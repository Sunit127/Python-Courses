"""Day 20: SQLite concurrency, WAL mode, and repository architecture.

Learning goals
--------------
1. Explain SQLite's single-writer rule and why connections should not be shared
   casually across threads.
2. Configure short-lived connections with WAL mode, a busy timeout, and foreign
   keys enabled.
3. Build a repository boundary that owns connection lifetime and keeps SQL out
   of the application layer.
4. Coordinate concurrent readers and writers safely with one connection per
   operation.
5. Make multi-row writes atomic and verify that a failed batch leaves no partial
   data behind.

Teaching notes
--------------
- SQLite allows many readers but serializes writes. WAL (write-ahead logging)
  lets readers continue while a writer appends to the log; it does not create
  unlimited writer parallelism.
- A sqlite3.Connection belongs to the thread that created it unless
  check_same_thread=False is used. Disabling that guard does not make a
  connection safe to share. Prefer one short-lived connection per repository
  operation or one connection per worker.
- Set a finite timeout (or PRAGMA busy_timeout) so lock contention becomes a
  controlled error instead of an infinite wait. Retrying blindly can duplicate
  a non-idempotent write.
- A repository is an application boundary: it translates domain values to
  parameterized SQL and rows back to typed values. Callers should not assemble
  SQL or manage commits.
- Use a transaction for a batch that must succeed or fail as a unit. The
  context manager commits on success and rolls back when an exception escapes.
- WAL is a file-database feature. An in-memory database cannot demonstrate a
  useful shared WAL journal, and each ":memory:" connection has its own data.
- A process with heavy write traffic may need a dedicated writer queue or a
  server database. WAL is a useful local coordination tool, not a replacement
  for capacity planning.

Run this file with Python 3.10+. It uses only the standard library and a
temporary SQLite file, so it is safe to execute repeatedly.

Practice exercises
------------------
1. Implement pending_titles(repository), returning pending task titles in
   case-insensitive alphabetical order. Keep the SQL inside the repository.
2. Implement read_consistent_snapshot(repository, workers). Run the same
   read-only query in several workers and return the sorted counts. Reject a
   non-positive worker count.
3. Add delete_completed() to TaskRepository. Delete completed rows in one
   transaction and return the number removed. Explain why a caller should not
   issue a separate SELECT and DELETE with a gap between them.

Solutions appear below the examples.

Expert challenge: production repository service
-----------------------------------------------
Design a small local service with a single writer queue and a pool of read
connections. Add schema migrations, idempotency keys for retried commands,
structured lock metrics, a backup command, and fault-injection tests. Prove
that a failed batch cannot leak partial rows, that readers see committed
snapshots, and that shutdown drains the writer queue. Document when SQLite
should be replaced by a server database.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Connection settings shared by repository operations."""

    path: str | Path
    timeout_seconds: float = 5.0
    wal: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        if isinstance(self.path, str) and not self.path.strip():
            raise ValueError("path cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(self.wal, bool):
            raise TypeError("wal must be a Boolean")


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """Immutable domain value returned by the repository."""

    task_id: int
    title: str
    completed: bool

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int):
            raise TypeError("task_id must be an integer")
        if self.task_id <= 0:
            raise ValueError("task_id must be positive")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be non-empty text")
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a Boolean")


def connect_database(config: DatabaseConfig) -> sqlite3.Connection:
    """Open and configure one independent connection.

    The caller owns the returned connection and must close it. Every repository
    method in this lesson follows that rule instead of sharing a global object.
    """
    if not isinstance(config, DatabaseConfig):
        raise TypeError("config must be a DatabaseConfig")

    connection = sqlite3.connect(
        str(config.path),
        timeout=config.timeout_seconds,
        isolation_level="DEFERRED",
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # The value is validated as a positive float before being converted.
    timeout_ms = max(1, int(config.timeout_seconds * 1000))
    connection.execute(f"PRAGMA busy_timeout = {timeout_ms}")

    if config.wal and str(config.path) != ":memory:":
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            connection.close()
            raise RuntimeError(f"could not enable WAL mode, got {mode!r}")

    return connection


def initialize_schema(config: DatabaseConfig) -> None:
    """Create the lesson schema idempotently."""
    connection = connect_database(config)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL COLLATE NOCASE UNIQUE,
                completed INTEGER NOT NULL DEFAULT 0
                    CHECK (completed IN (0, 1))
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_pending_title
                ON tasks (completed, title COLLATE NOCASE);
            """
        )
        connection.commit()
    finally:
        connection.close()


def _clean_title(value: str) -> str:
    """Normalize a title at the domain boundary."""
    if not isinstance(value, str):
        raise TypeError("title must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("title cannot be empty")
    return cleaned


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    """Validate and convert one database row."""
    raw_id = row["id"]
    raw_title = row["title"]
    raw_completed = row["completed"]
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise TypeError("database id must be an integer")
    if not isinstance(raw_title, str):
        raise TypeError("database title must be text")
    if raw_completed not in (0, 1):
        raise ValueError("database completed flag must be 0 or 1")
    return TaskRecord(
        task_id=raw_id,
        title=raw_title,
        completed=bool(raw_completed),
    )


class TaskRepository:
    """A small repository whose methods each own one SQLite connection.

    The repository object is safe to share between threads because it stores
    configuration, not a live connection. Each operation gets a fresh
    connection, performs a short transaction or read, and closes it.
    """

    def __init__(self, config: DatabaseConfig):
        if not isinstance(config, DatabaseConfig):
            raise TypeError("config must be a DatabaseConfig")
        self._config = config

    @property
    def config(self) -> DatabaseConfig:
        """Expose immutable configuration without exposing a connection."""
        return self._config

    def initialize(self) -> None:
        initialize_schema(self._config)

    def journal_mode(self) -> str:
        """Return the active journal mode for a diagnostic check."""
        connection = connect_database(self._config)
        try:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        finally:
            connection.close()

    def add_task(self, title: str) -> TaskRecord:
        """Insert one task and translate expected uniqueness failures."""
        cleaned_title = _clean_title(title)
        connection = connect_database(self._config)
        try:
            try:
                with connection:
                    cursor = connection.execute(
                        "INSERT INTO tasks (title, completed) VALUES (?, ?)",
                        (cleaned_title, 0),
                    )
                    row = connection.execute(
                        "SELECT id, title, completed FROM tasks WHERE id = ?",
                        (cursor.lastrowid,),
                    ).fetchone()
            except sqlite3.IntegrityError as error:
                raise ValueError(f"task title already exists: {cleaned_title!r}") from error
            if row is None:
                raise RuntimeError("inserted task could not be read back")
            return _row_to_task(row)
        finally:
            connection.close()

    def add_many(self, titles: Iterable[str]) -> tuple[TaskRecord, ...]:
        """Insert a batch atomically; a duplicate rolls the entire batch back."""
        materialized = tuple(_clean_title(title) for title in titles)
        if not materialized:
            raise ValueError("provide at least one title")

        connection = connect_database(self._config)
        inserted_ids: list[int] = []
        try:
            try:
                with connection:
                    for title in materialized:
                        cursor = connection.execute(
                            "INSERT INTO tasks (title, completed) VALUES (?, ?)",
                            (title, 0),
                        )
                        inserted_ids.append(int(cursor.lastrowid))
            except sqlite3.IntegrityError as error:
                raise ValueError("batch rolled back because a title conflicted") from error
            return tuple(self.get_task(task_id) for task_id in inserted_ids)
        finally:
            connection.close()

    def get_task(self, task_id: int) -> TaskRecord:
        """Read one task or raise LookupError."""
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise TypeError("task_id must be an integer")

        connection = connect_database(self._config)
        try:
            row = connection.execute(
                "SELECT id, title, completed FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"unknown task id: {task_id}")
            return _row_to_task(row)
        finally:
            connection.close()

    def list_tasks(self, *, include_completed: bool = True) -> tuple[TaskRecord, ...]:
        """Read tasks in deterministic title order."""
        if not isinstance(include_completed, bool):
            raise TypeError("include_completed must be a Boolean")

        connection = connect_database(self._config)
        try:
            query = (
                "SELECT id, title, completed FROM tasks"
                " ORDER BY title COLLATE NOCASE, id"
            )
            parameters: tuple[object, ...] = ()
            if not include_completed:
                query = (
                    "SELECT id, title, completed FROM tasks"
                    " WHERE completed = ? ORDER BY title COLLATE NOCASE, id"
                )
                parameters = (0,)
            rows = connection.execute(query, parameters)
            return tuple(_row_to_task(row) for row in rows)
        finally:
            connection.close()

    def pending_titles(self) -> tuple[str, ...]:
        """Exercise 1 solution: keep the query in the repository."""
        connection = connect_database(self._config)
        try:
            rows = connection.execute(
                """
                SELECT title
                FROM tasks
                WHERE completed = ?
                ORDER BY title COLLATE NOCASE, id
                """,
                (0,),
            )
            return tuple(str(row["title"]) for row in rows)
        finally:
            connection.close()

    def complete_task(self, task_id: int) -> TaskRecord:
        """Mark one task complete atomically and return its new value."""
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise TypeError("task_id must be an integer")

        connection = connect_database(self._config)
        try:
            with connection:
                cursor = connection.execute(
                    "UPDATE tasks SET completed = ? WHERE id = ?",
                    (1, task_id),
                )
                if cursor.rowcount != 1:
                    raise LookupError(f"unknown task id: {task_id}")
            return self.get_task(task_id)
        finally:
            connection.close()

    def pending_count(self) -> int:
        """Return one committed read-only result."""
        connection = connect_database(self._config)
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE completed = ?",
                (0,),
            ).fetchone()
            if row is None:
                raise RuntimeError("count query returned no row")
            return int(row["count"])
        finally:
            connection.close()

    def delete_completed(self) -> int:
        """Exercise 3 solution: delete completed rows in one transaction."""
        connection = connect_database(self._config)
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM tasks WHERE completed = ?",
                    (1,),
                )
                return int(cursor.rowcount)
        finally:
            connection.close()


def read_consistent_snapshot(
    repository: TaskRepository,
    workers: int = 4,
) -> tuple[int, ...]:
    """Exercise 2 solution: run independent read operations concurrently."""
    if not isinstance(repository, TaskRepository):
        raise TypeError("repository must be a TaskRepository")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers <= 0:
        raise ValueError("workers must be positive")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(repository.pending_count) for _ in range(workers)]
        return tuple(sorted(future.result() for future in futures))


def add_in_parallel(
    repository: TaskRepository,
    titles: Sequence[str],
    *,
    workers: int = 4,
) -> tuple[TaskRecord, ...]:
    """Submit independent inserts; SQLite serializes the short write sections."""
    if not isinstance(repository, TaskRepository):
        raise TypeError("repository must be a TaskRepository")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers <= 0:
        raise ValueError("workers must be positive")
    materialized = tuple(titles)
    if not materialized:
        raise ValueError("provide at least one title")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(repository.add_task, title) for title in materialized]
        records = tuple(future.result() for future in futures)
    return tuple(sorted(records, key=lambda record: record.task_id))


def run_self_checks() -> None:
    """Exercise normal behavior, rollback, concurrency, and edge cases."""
    with TemporaryDirectory() as directory:
        config = DatabaseConfig(Path(directory) / "checks.sqlite3")
        repository = TaskRepository(config)
        repository.initialize()

        assert repository.journal_mode() == "wal"
        first = repository.add_task("  Read WAL documentation  ")
        second = repository.add_task("Repository boundary")
        assert first.title == "Read WAL documentation"
        assert repository.pending_count() == 2
        assert repository.pending_titles() == (
            "Read WAL documentation",
            "Repository boundary",
        )
        assert read_consistent_snapshot(repository, workers=5) == (2, 2, 2, 2, 2)

        completed = repository.complete_task(first.task_id)
        assert completed.completed
        assert repository.pending_count() == 1
        assert repository.list_tasks(include_completed=False) == (second,)

        inserted = repository.add_many(("Atomic one", "Atomic two"))
        assert [record.title for record in inserted] == ["Atomic one", "Atomic two"]
        before_failed_batch = repository.list_tasks()
        try:
            repository.add_many(("Never committed", "Atomic one"))
        except ValueError as error:
            assert "rolled back" in str(error)
        else:
            raise AssertionError("duplicate batch must fail")
        assert repository.list_tasks() == before_failed_batch
        assert "Never committed" not in {record.title for record in repository.list_tasks()}

        parallel_records = add_in_parallel(
            repository,
            tuple(f"Concurrent {number}" for number in range(8)),
            workers=4,
        )
        assert len(parallel_records) == 8
        assert len({record.task_id for record in parallel_records}) == 8
        assert repository.pending_count() == 11

        removed = repository.delete_completed()
        assert removed == 1
        assert all(not record.completed for record in repository.list_tasks())

        invalid_calls = (
            lambda: DatabaseConfig("x", timeout_seconds=0),
            lambda: repository.add_task(""),
            lambda: repository.add_task("Repository boundary"),
            lambda: repository.get_task(999_999),
            lambda: repository.list_tasks(include_completed="yes"),  # type: ignore[arg-type]
            lambda: read_consistent_snapshot(repository, workers=0),
            lambda: add_in_parallel(repository, (), workers=2),
            lambda: TaskRecord(0, "bad", False),
        )
        for invalid_call in invalid_calls:
            try:
                invalid_call()
            except (LookupError, TypeError, ValueError):
                pass
            else:
                raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a deterministic demo against a temporary file database."""
    run_self_checks()

    with TemporaryDirectory() as directory:
        config = DatabaseConfig(Path(directory) / "demo.sqlite3")
        repository = TaskRepository(config)
        repository.initialize()
        repository.add_many(("Design the repository", "Measure lock waits"))
        print("Journal mode:", repository.journal_mode())
        print("Initial pending titles:", repository.pending_titles())
        print("Concurrent snapshots:", read_consistent_snapshot(repository, workers=3))

        added = add_in_parallel(
            repository,
            ("Reader one", "Reader two", "Writer queue"),
            workers=3,
        )
        print("Parallel inserts:", added)
        repository.complete_task(added[0].task_id)
        print("Remaining pending:", repository.pending_titles())
        print("Deleted completed rows:", repository.delete_completed())

    print("Self-checks passed.")


if __name__ == "__main__":
    main()
