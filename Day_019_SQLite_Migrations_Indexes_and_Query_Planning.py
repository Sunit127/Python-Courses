"""Day 19: SQLite migrations, indexes, and query planning.

Learning goals
--------------
1. Version a SQLite schema with small, repeatable migrations.
2. Apply migrations transactionally so failed upgrades roll back cleanly.
3. Design a composite index around a real WHERE and ORDER BY query.
4. Read EXPLAIN QUERY PLAN output and validate index usage.
5. Keep schema changes, repository code, and validation responsibilities clear.

Teaching notes
--------------
- A migration is an ordered change from one known schema version to the next.
  Keep migrations append-only once a database may have run them.
- SQLite stores an application version in PRAGMA user_version. It is an integer,
  not a replacement for documenting the actual schema.
- Start a migration transaction explicitly. Commit DDL and the version bump
  together; rollback both when any step fails.
- An index speeds reads that match its leading columns, but it consumes space
  and makes inserts and updates more expensive.
- For WHERE completed = 0 ORDER BY priority DESC, created_at ASC, an index
  beginning with (completed, priority, created_at) can avoid extra work.
- EXPLAIN QUERY PLAN is a diagnostic, not a performance guarantee. Benchmark
  realistic data before adding an index just because a plan looks convenient.
- Values can be parameterized. SQL identifiers cannot; whitelist any dynamic
  column choice before composing a query.
- A migration runner should be idempotent: a current database is unchanged.

Run with Python 3.10+. Only standard-library sqlite3 and an in-memory database
are required.

Practice exercises
------------------
1. Implement parse_due_date(value). Accept YYYY-MM-DD and return datetime.date.
2. Implement pending_titles(connection), ordering dated tasks before undated
   tasks and using priority as the primary ordering.
3. Extend explain_pending_query so it reports whether a table scan appears.
   Explain why scanning a tiny table can still be the best plan.

Solutions appear below the examples.

Expert challenge: production migration runner
---------------------------------------------
Design versions 1 through 4. Add an audit_events table, backfill a derived
value in batches, add a dry-run mode, and enforce a single-writer policy.
Test a fresh database, an old database with rows, an already-current database,
a failed migration rollback, and a reopened file database. Capture migration
duration and redact secrets from diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
import sqlite3
from typing import TypeAlias


DatabaseTarget: TypeAlias = str


@dataclass(frozen=True, slots=True)
class Task:
    """Immutable application value read from SQLite."""

    task_id: int
    title: str
    priority: int
    completed: bool
    created_at: str
    due_date: date | None


def _clean_title(value: str) -> str:
    """Normalize a title at the application boundary."""
    if not isinstance(value, str):
        raise TypeError("title must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("title cannot be empty")
    return cleaned


def _priority(value: int) -> int:
    """Validate the schema's five-level priority."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("priority must be an integer")
    if not 1 <= value <= 5:
        raise ValueError("priority must be between 1 and 5")
    return value


def parse_due_date(value: str | None) -> date | None:
    """Convert an ISO date string to a date, allowing an absent deadline."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("due date must be text or None")
    if value.strip() == "":
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError("due date must use YYYY-MM-DD") from error


def _due_date_text(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def connect_database(database: DatabaseTarget = ":memory:") -> sqlite3.Connection:
    """Open a connection configured for named row access."""
    if not isinstance(database, str):
        raise TypeError("database must be a file path or ':memory:'")
    connection = sqlite3.connect(database, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def schema_version(connection: sqlite3.Connection) -> int:
    """Read the application-managed SQLite schema version."""
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None:
        raise RuntimeError("SQLite did not return user_version")
    return int(row[0])


def _migration_1(connection: sqlite3.Connection) -> None:
    """Create the initial table used by the previous lesson."""
    connection.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL COLLATE NOCASE UNIQUE,
            priority INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 5),
            completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
            created_at TEXT NOT NULL
        )
        """
    )


def _migration_2(connection: sqlite3.Connection) -> None:
    """Add deadlines and the composite index used by the main query."""
    connection.execute("ALTER TABLE tasks ADD COLUMN due_date TEXT")
    connection.execute(
        """
        CREATE INDEX idx_tasks_pending_priority_created
        ON tasks (completed, priority DESC, created_at ASC)
        """
    )


def migrate(connection: sqlite3.Connection) -> int:
    """Upgrade a connection to version 2 or return its current version."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3.Connection")
    if connection.in_transaction:
        raise RuntimeError("migrate requires an idle connection")

    current = schema_version(connection)
    if current > 2:
        raise RuntimeError(f"unsupported future schema version: {current}")
    if current == 2:
        return 2

    # BEGIN is explicit because a migration must include transactional DDL.
    connection.execute("BEGIN")
    try:
        if current < 1:
            _migration_1(connection)
            connection.execute("PRAGMA user_version = 1")
            current = 1
        if current < 2:
            _migration_2(connection)
            connection.execute("PRAGMA user_version = 2")
            current = 2
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
    return current


def _row_to_task(row: sqlite3.Row) -> Task:
    """Convert one row while checking the database boundary."""
    raw_due_date = row["due_date"]
    if raw_due_date is not None and not isinstance(raw_due_date, str):
        raise TypeError("database due_date must be text or NULL")
    return Task(
        task_id=int(row["id"]),
        title=_clean_title(row["title"]),
        priority=_priority(int(row["priority"])),
        completed=bool(row["completed"]),
        created_at=str(row["created_at"]),
        due_date=parse_due_date(raw_due_date),
    )


def add_task(
    connection: sqlite3.Connection,
    title: str,
    *,
    priority: int = 3,
    completed: bool = False,
    created_at: str = "2026-09-23T00:00:00+00:00",
    due_date: date | None = None,
) -> Task:
    """Insert one validated task and return the stored value."""
    cleaned_title = _clean_title(title)
    normalized_priority = _priority(priority)
    if not isinstance(completed, bool):
        raise TypeError("completed must be a Boolean")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("created_at must be non-empty text")
    if due_date is not None and not isinstance(due_date, date):
        raise TypeError("due_date must be a date or None")

    with connection:
        cursor = connection.execute(
            """
            INSERT INTO tasks (title, priority, completed, created_at, due_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cleaned_title,
                normalized_priority,
                int(completed),
                created_at,
                _due_date_text(due_date),
            ),
        )
        row = connection.execute(
            """
            SELECT id, title, priority, completed, created_at, due_date
            FROM tasks WHERE id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()
    if row is None:
        raise RuntimeError("inserted task could not be read back")
    return _row_to_task(row)


def pending_tasks(connection: sqlite3.Connection) -> tuple[Task, ...]:
    """Return pending tasks in the order supported by the composite index."""
    rows = connection.execute(
        """
        SELECT id, title, priority, completed, created_at, due_date
        FROM tasks
        WHERE completed = ?
        ORDER BY priority DESC, created_at ASC
        """,
        (0,),
    )
    return tuple(_row_to_task(row) for row in rows)


def pending_titles(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Solution: order dated tasks first and undated tasks last."""
    rows = connection.execute(
        """
        SELECT title
        FROM tasks
        WHERE completed = ?
        ORDER BY priority DESC, due_date IS NULL, due_date ASC,
                 title COLLATE NOCASE
        """,
        (0,),
    )
    return tuple(str(row["title"]) for row in rows)


def explain_pending_query(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Return SQLite's human-readable plan details."""
    rows = connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT id, title, priority, completed, created_at, due_date
        FROM tasks
        WHERE completed = 0
        ORDER BY priority DESC, created_at ASC
        """
    )
    return tuple(str(row[3]) for row in rows)


def plan_uses_pending_index(connection: sqlite3.Connection) -> bool:
    """Check the plan without depending on numeric detail fields."""
    return any(
        "idx_tasks_pending_priority_created" in detail
        for detail in explain_pending_query(connection)
    )


def safe_order_clause(column: str) -> str:
    """Whitelist an identifier-like choice before composing SQL."""
    allowed = {"priority": "priority DESC", "created_at": "created_at ASC"}
    try:
        return allowed[column]
    except KeyError as error:
        raise ValueError("unsupported sort column") from error


def list_tasks(
    connection: sqlite3.Connection,
    *,
    sort_by: str = "priority",
) -> tuple[Task, ...]:
    """Demonstrate safe dynamic SQL for a whitelisted sort choice."""
    order_clause = safe_order_clause(sort_by)
    rows = connection.execute(
        f"""
        SELECT id, title, priority, completed, created_at, due_date
        FROM tasks ORDER BY {order_clause}
        """
    )
    return tuple(_row_to_task(row) for row in rows)


@contextmanager
def migrated_session(database: DatabaseTarget = ":memory:"):
    """Open, migrate, and close a database for one short-lived operation."""
    connection = connect_database(database)
    try:
        migrate(connection)
        yield connection
    finally:
        connection.close()


def _legacy_connection() -> sqlite3.Connection:
    """Create a version-1 database to demonstrate an upgrade."""
    connection = connect_database()
    connection.execute("BEGIN")
    try:
        _migration_1(connection)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO tasks (title, priority, completed, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("Legacy row", 4, 0, "2026-09-22T10:00:00+00:00"),
        )
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
    return connection


def run_self_checks() -> None:
    """Exercise migrations, indexes, queries, and edge cases."""
    with migrated_session() as connection:
        assert schema_version(connection) == 2
        assert migrate(connection) == 2
        assert "due_date" in {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(tasks)")
        }

        legacy = _legacy_connection()
        try:
            assert schema_version(legacy) == 1
            assert migrate(legacy) == 2
            legacy_tasks = pending_tasks(legacy)
            assert [task.title for task in legacy_tasks] == ["Legacy row"]
            assert legacy_tasks[0].due_date is None
            assert plan_uses_pending_index(legacy)
        finally:
            legacy.close()

        first = add_task(
            connection,
            "Index design",
            priority=5,
            created_at="2026-09-23T08:00:00+00:00",
            due_date=date(2026, 9, 30),
        )
        second = add_task(
            connection,
            "Read the plan",
            priority=4,
            created_at="2026-09-23T08:01:00+00:00",
        )
        completed = add_task(
            connection,
            "Already done",
            priority=5,
            completed=True,
            created_at="2026-09-23T08:02:00+00:00",
        )
        assert first.due_date == date(2026, 9, 30)
        assert [task.title for task in pending_tasks(connection)] == [
            "Index design",
            "Read the plan",
        ]
        assert pending_titles(connection) == ("Index design", "Read the plan")
        assert plan_uses_pending_index(connection)
        assert any(
            "SEARCH tasks USING INDEX" in detail
            for detail in explain_pending_query(connection)
        )
        assert list_tasks(connection, sort_by="created_at")[0].title == "Index design"

        invalid_calls: Iterable[object] = (
            lambda: parse_due_date("2026-02-30"),
            lambda: parse_due_date("tomorrow"),
            lambda: parse_due_date(20260923),  # type: ignore[arg-type]
            lambda: add_task(connection, "bad priority", priority=0),
            lambda: safe_order_clause("title"),
        )
        for invalid_call in invalid_calls:
            try:
                invalid_call()  # type: ignore[operator]
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("invalid input must raise a specific error")

        assert completed.completed
        assert pending_titles(connection) == ("Index design", "Read the plan")

    fresh = connect_database()
    try:
        assert migrate(fresh) == 2
        assert migrate(fresh) == 2
    finally:
        fresh.close()


def main() -> None:
    """Run safe demonstrations and print the query planner's explanation."""
    run_self_checks()
    with migrated_session() as connection:
        add_task(connection, "Ship migration", priority=5, due_date=date(2026, 10, 1))
        add_task(connection, "Benchmark realistic data", priority=2)
        print("Pending:", pending_titles(connection))
        print("Query plan:")
        for detail in explain_pending_query(connection):
            print(" -", detail)
        print("Uses composite index:", plan_uses_pending_index(connection))
        print("Self-checks passed.")


if __name__ == "__main__":
    main()
