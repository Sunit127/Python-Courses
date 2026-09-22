"""Day 18: SQLite fundamentals, transactions, and safe persistence.

Learning goals
--------------
1. Create a small SQLite schema with constraints that protect data.
2. Use parameterized SQL for every value supplied by a caller.
3. Separate database effects from pure parsing and domain validation.
4. Commit related writes atomically and understand rollback behavior.
5. Read rows into typed, immutable domain objects and close connections safely.

Teaching notes
--------------
- SQLite is a relational database stored in one file (or in memory). A table
  stores rows, columns describe values, and a primary key identifies a row.
- Define constraints in the schema as well as in Python. NOT NULL, CHECK, and
  UNIQUE rules protect data even when another program writes to the database.
- Never build SQL by concatenating user text. Use ? placeholders and pass a
  separate tuple of values to execute().
- A sqlite3.Connection is a context manager. "with connection:" commits when
  the block succeeds and rolls back when an exception escapes it.
- Transactions should cover one business operation. A bulk import should not
  leave half its rows committed after one malformed or duplicate row.
- sqlite3.Row gives named column access, but a domain object is a better public
  boundary than leaking database details throughout an application.
- Keep connections short-lived in small scripts and inject them into functions
  so tests can use ":memory:". A production service should decide connection
  ownership, timeouts, backups, and migrations explicitly.
- Parameterized queries prevent SQL injection, but authorization and validation
  are still application responsibilities. Do not store passwords in plaintext.
- SQLite is excellent for local tools and moderate workloads. When many writers,
  replicas, or operational guarantees are required, choose a server database
  deliberately.

Run this file with Python 3.10+ to execute deterministic examples and checks.
It uses only the standard-library sqlite3 module and an in-memory database.

Practice exercises
------------------
1. Implement parse_task_lines(lines). Parse non-empty "title|priority" lines,
   reject malformed lines with their line number, and return immutable tuples.
2. Implement find_tasks(connection, phrase). Search titles case-insensitively with a parameterized query, preserving
   priority then title ordering.
3. Implement completed_ratio(tasks). Return 0.0 for an empty iterable and a
   rounded fraction for a non-empty iterable.

Solutions appear below the examples.

Expert challenge: durable task database
----------------------------------------
Extend TaskRepository with schema migrations and an audit_log table. Add a
version table, migrate version 1 to version 2 without losing rows, and record
one audit event for every state-changing operation. Make the migration
transactional, redact sensitive text in diagnostics, and add tests for a
fresh database, an already-migrated database, a failed migration, and a
reopened file database. Keep SQL values parameterized and keep connection
ownership explicit.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import TypeAlias


DatabaseTarget: TypeAlias = str | Path


def _clean_title(value: str) -> str:
    """Return a non-empty title with stable whitespace."""
    if not isinstance(value, str):
        raise TypeError("title must be a string")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("title cannot be empty")
    return cleaned


def _priority(value: int) -> int:
    """Validate the five-level priority used by the schema."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("priority must be an integer")
    if not 1 <= value <= 5:
        raise ValueError("priority must be between 1 and 5")
    return value


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """An immutable task value returned by the repository."""

    task_id: int
    title: str
    priority: int
    completed: bool
    created_at: str

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int):
            raise TypeError("task_id must be an integer")
        if self.task_id <= 0:
            raise ValueError("task_id must be positive")
        object.__setattr__(self, "title", _clean_title(self.title))
        object.__setattr__(self, "priority", _priority(self.priority))
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a Boolean")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be non-empty text")


def _row_to_task(row: Mapping[str, object]) -> TaskRecord:
    """Convert one sqlite row while checking the database boundary."""
    required = {"id", "title", "priority", "completed", "created_at"}
    if set(row.keys()) != required:
        raise ValueError("task row has an unexpected shape")

    raw_id = row["id"]
    raw_title = row["title"]
    raw_priority = row["priority"]
    raw_completed = row["completed"]
    raw_created_at = row["created_at"]

    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise TypeError("database id must be an integer")
    if not isinstance(raw_title, str):
        raise TypeError("database title must be text")
    if isinstance(raw_priority, bool) or not isinstance(raw_priority, int):
        raise TypeError("database priority must be an integer")
    if raw_completed not in (0, 1):
        raise ValueError("database completed flag must be 0 or 1")
    if not isinstance(raw_created_at, str):
        raise TypeError("database created_at must be text")

    return TaskRecord(
        task_id=raw_id,
        title=raw_title,
        priority=raw_priority,
        completed=bool(raw_completed),
        created_at=raw_created_at,
    )


def connect_database(database: DatabaseTarget = ":memory:") -> sqlite3.Connection:
    """Open a configured connection without creating application tables."""
    if not isinstance(database, (str, Path)):
        raise TypeError("database must be a path or ':memory:'")

    connection = sqlite3.connect(str(database), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the task table idempotently."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL COLLATE NOCASE UNIQUE,
            priority INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 5),
            completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
            created_at TEXT NOT NULL
        );
        """
    )
    connection.commit()


@contextmanager
def database_session(
    database: DatabaseTarget = ":memory:",
) -> Iterator[sqlite3.Connection]:
    """Own a connection for one scope and always close it."""
    connection = connect_database(database)
    try:
        initialize_schema(connection)
        yield connection
    finally:
        connection.close()


def _now_text() -> str:
    """Use a sortable UTC timestamp without requiring a custom adapter."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_task(
    connection: sqlite3.Connection,
    title: str,
    *,
    priority: int = 3,
    created_at: str | None = None,
) -> TaskRecord:
    """Insert one task and return it, translating expected conflicts."""
    cleaned_title = _clean_title(title)
    normalized_priority = _priority(priority)
    timestamp = _now_text() if created_at is None else created_at
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("created_at must be non-empty text")

    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO tasks (title, priority, completed, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cleaned_title, normalized_priority, 0, timestamp),
            )
            row = connection.execute(
                """
                SELECT id, title, priority, completed, created_at
                FROM tasks
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
    except sqlite3.IntegrityError as error:
        raise ValueError(f"task title already exists: {cleaned_title!r}") from error

    if row is None:
        raise RuntimeError("inserted task could not be read back")
    return _row_to_task(row)


def get_task(
    connection: sqlite3.Connection,
    task_id: int,
) -> TaskRecord:
    """Return one task or raise LookupError."""
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise TypeError("task_id must be an integer")

    row = connection.execute(
        """
        SELECT id, title, priority, completed, created_at
        FROM tasks
        WHERE id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"unknown task id: {task_id}")
    return _row_to_task(row)


def list_tasks(
    connection: sqlite3.Connection,
    *,
    include_completed: bool = True,
) -> tuple[TaskRecord, ...]:
    """Return tasks in deterministic priority/title order."""
    if not isinstance(include_completed, bool):
        raise TypeError("include_completed must be a Boolean")

    query = """
        SELECT id, title, priority, completed, created_at
        FROM tasks
    """
    parameters: tuple[object, ...] = ()
    if not include_completed:
        query += " WHERE completed = ?"
        parameters = (0,)
    query += " ORDER BY priority DESC, title COLLATE NOCASE, id"

    rows = connection.execute(query, parameters)
    return tuple(_row_to_task(row) for row in rows)


def complete_task(
    connection: sqlite3.Connection,
    task_id: int,
) -> TaskRecord:
    """Mark a task complete in one atomic write and read."""
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise TypeError("task_id must be an integer")

    with connection:
        cursor = connection.execute(
            "UPDATE tasks SET completed = ? WHERE id = ?",
            (1, task_id),
        )
        if cursor.rowcount != 1:
            raise LookupError(f"unknown task id: {task_id}")
    return get_task(connection, task_id)


def add_tasks_atomically(
    connection: sqlite3.Connection,
    tasks: Iterable[tuple[str, int]],
) -> tuple[TaskRecord, ...]:
    """Insert all tasks or none, demonstrating transaction rollback."""
    materialized = tuple(tasks)
    if not materialized:
        raise ValueError("provide at least one task")

    validated = tuple(
        (_clean_title(title), _priority(priority))
        for title, priority in materialized
    )

    inserted_ids: list[int] = []
    try:
        with connection:
            for title, priority in validated:
                cursor = connection.execute(
                    """
                    INSERT INTO tasks (title, priority, completed, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (title, priority, 0, _now_text()),
                )
                inserted_ids.append(int(cursor.lastrowid))
    except sqlite3.IntegrityError as error:
        raise ValueError("bulk insert rolled back because a title conflicted") from error

    return tuple(get_task(connection, task_id) for task_id in inserted_ids)


def parse_task_lines(
    lines: Iterable[str],
) -> tuple[tuple[str, int], ...]:
    """Solution 1: parse title|priority input before opening a transaction."""
    parsed: list[tuple[str, int]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not isinstance(raw_line, str):
            raise TypeError(f"line {line_number} must be text")
        line = raw_line.strip()
        if not line:
            continue

        title_text, separator, priority_text = line.partition("|")
        if not separator:
            raise ValueError(f"line {line_number}: expected title|priority")
        try:
            priority_value = int(priority_text.strip())
        except ValueError as error:
            raise ValueError(
                f"line {line_number}: priority must be an integer"
            ) from error
        parsed.append((_clean_title(title_text), _priority(priority_value)))
    return tuple(parsed)


def find_tasks(
    connection: sqlite3.Connection,
    phrase: str,
) -> tuple[TaskRecord, ...]:
    """Solution 2: search with a parameterized, literal substring query."""
    cleaned_phrase = _clean_title(phrase)
    rows = connection.execute(
        """
        SELECT id, title, priority, completed, created_at
        FROM tasks
        WHERE instr(lower(title), lower(?)) > 0
        ORDER BY priority DESC, title COLLATE NOCASE, id
        """,
        (cleaned_phrase,),
    )
    return tuple(_row_to_task(row) for row in rows)


def completed_ratio(tasks: Iterable[TaskRecord]) -> float:
    """Solution 3: calculate a rounded ratio from any iterable."""
    materialized = tuple(tasks)
    if not all(isinstance(task, TaskRecord) for task in materialized):
        raise TypeError("tasks must contain TaskRecord values")
    if not materialized:
        return 0.0
    completed_count = sum(task.completed for task in materialized)
    return round(completed_count / len(materialized), 2)


def run_self_checks() -> None:
    """Check schema constraints, transactions, queries, and edge cases."""
    with database_session() as connection:
        first = add_task(connection, " Parameterized SQL ", priority=5, created_at="2026-09-22T09:00:00+00:00")
        second = add_task(connection, "Transactions", priority=4, created_at="2026-09-22T09:00:01+00:00")
        assert first.title == "Parameterized SQL"
        assert first.priority == 5
        assert list_tasks(connection) == (first, second)
        assert completed_ratio(list_tasks(connection)) == 0.0

        completed = complete_task(connection, first.task_id)
        assert completed.completed
        assert list_tasks(connection, include_completed=False) == (second,)

        assert find_tasks(connection, "sql") == (completed,)
        assert find_tasks(connection, "%") == ()
        assert parse_task_lines(
            ["Queries|5", "", " Transactions | 4 "]
        ) == (("Queries", 5), ("Transactions", 4))

        bulk = add_tasks_atomically(
            connection,
            [("Indexes", 2), ("Testing", 3)],
        )
        assert [task.title for task in bulk] == ["Indexes", "Testing"]
        before_rollback = list_tasks(connection)
        try:
            add_tasks_atomically(connection, [("Committed", 1), ("Testing", 4)])
        except ValueError as error:
            assert "rolled back" in str(error)
        else:
            raise AssertionError("duplicate bulk insert must fail")
        assert list_tasks(connection) == before_rollback

        invalid_calls = (
            lambda: add_task(connection, ""),
            lambda: add_task(connection, "bad priority", priority=6),
            lambda: add_task(connection, "Parameterized SQL"),
            lambda: get_task(connection, 999),
            lambda: complete_task(connection, 999),
            lambda: add_tasks_atomically(connection, []),
            lambda: parse_task_lines(["missing-priority"]),
            lambda: parse_task_lines(["bad|zero", "x|1"]),
            lambda: find_tasks(connection, ""),
            lambda: completed_ratio([object()]),  # type: ignore[list-item]
        )
        for invalid_call in invalid_calls:
            try:
                invalid_call()
            except (LookupError, TypeError, ValueError):
                pass
            else:
                raise AssertionError("invalid input must raise a specific error")

        # The failed duplicate did not leave its first row behind.
        assert "Committed" not in {task.title for task in list_tasks(connection)}

    try:
        connect_database(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("invalid database target must raise TypeError")


def main() -> None:
    """Run a safe, deterministic SQLite demonstration."""
    run_self_checks()

    with database_session() as connection:
        add_tasks_atomically(
            connection,
            [
                ("Read the schema", 4),
                ("Practice transactions", 5),
                ("Write a query", 3),
            ],
        )
        first = list_tasks(connection)[0]
        complete_task(connection, first.task_id)
        print("All tasks:", list_tasks(connection))
        print("Pending tasks:", list_tasks(connection, include_completed=False))
        print("Matching 'query':", find_tasks(connection, "query"))
        print("Completed ratio:", completed_ratio(list_tasks(connection)))

    print("Self-checks passed.")


if __name__ == "__main__":
    main()
