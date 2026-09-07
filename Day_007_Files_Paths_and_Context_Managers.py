"""Day 7: Files, paths, and context managers.

Learning goals
--------------
1. Represent filesystem locations with pathlib.Path instead of fragile strings.
2. Read and write UTF-8 text deliberately, including newline behavior.
3. Use context managers so files and temporary resources are always released.
4. Separate filesystem effects from pure parsing and reporting logic.
5. Handle missing paths, malformed data, and edge cases with useful errors.

Teaching notes
--------------
- A Path is an object that can join, inspect, and transform locations. Use
  Path("reports") / "daily.txt" instead of manually concatenating slashes.
- resolve() can consult the filesystem; absolute() only makes a path absolute.
  Do not resolve untrusted paths and then assume they are safe: validate that
  a candidate stays inside an approved directory.
- open(path, mode, encoding="utf-8", newline="") returns a resource-owning
  context manager. The with block closes it even when an exception occurs.
- "r" reads, "w" replaces, "a" appends, and "x" creates only when absent.
  Prefer explicit encodings and avoid accidental overwrites.
- File APIs raise specific exceptions such as FileNotFoundError,
  PermissionError, and IsADirectoryError. Catch only errors you can handle.
- A context manager may be a class with __enter__/__exit__, or a generator
  decorated with contextlib.contextmanager. Keep cleanup in one place.
- Keep parsing functions pure where possible. A function that receives text is
  easier to test than one that opens a file, parses it, and prints at once.
- TemporaryDirectory is ideal for tests and demonstrations: it cleans up its
  directory automatically when the context exits.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Write read_nonempty_lines(path) that returns stripped, non-empty lines and
   raises FileNotFoundError naturally.
2. Write safe_child_path(root, relative_name). Reject absolute names and any
   path that would escape root after normalization.
3. Write summarize_numbers_file(path) that parses one finite float per line and
   returns count, total, and average. Report the line number for bad input.

Solutions and clear guidance appear below the examples.

Expert challenge: durable report exporter
-----------------------------------------
Build an export_report(report, destination) function that writes a small,
human-readable report without partially overwriting an existing file. Validate
the destination's parent, write to a temporary sibling in the same directory,
flush and close it, then replace the destination with os.replace. Add tests for
an empty report, an existing destination, and a failed write. Keep all temporary
files inside the approved directory and never accept a user-controlled absolute
path without an explicit policy.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


def write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text and return the Path that was written.

    Parent directories must already exist; silently creating arbitrary
    directories is often a surprising side effect for a library function.
    """
    target = Path(path)
    if not target.name:
        raise ValueError("path must name a file")
    target.write_text(text, encoding="utf-8", newline="")
    return target


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file while preserving its contents."""
    return Path(path).read_text(encoding="utf-8")


def nonempty_lines(text: str) -> list[str]:
    """Pure parser: strip whitespace and discard blank lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_nonempty_lines(path: str | Path) -> list[str]:
    """Solution 1: combine a file effect with the pure line parser."""
    return nonempty_lines(read_text(path))


def safe_child_path(root: str | Path, relative_name: str | Path) -> Path:
    """Solution 2: return a child of root without allowing traversal escape."""
    base = Path(root).resolve()
    candidate = Path(relative_name)
    if candidate.is_absolute():
        raise ValueError("relative_name must not be absolute")

    target = (base / candidate).resolve()
    try:
        target.relative_to(base)
    except ValueError as error:
        raise ValueError("relative_name escapes the approved root") from error
    return target


def summarize_number_text(text: str) -> dict[str, float]:
    """Pure parser for one finite number per non-empty line."""
    values: list[float] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError as error:
            raise ValueError(
                f"line {line_number} is not a number: {stripped!r}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"line {line_number} must contain a finite number")
        values.append(value)

    if not values:
        raise ValueError("the input contains no numbers")

    total = sum(values)
    return {
        "count": float(len(values)),
        "total": round(total, 2),
        "average": round(total / len(values), 2),
    }


def summarize_numbers_file(path: str | Path) -> dict[str, float]:
    """Solution 3: read once, then delegate parsing to a pure function."""
    return summarize_number_text(read_text(path))


def write_json(path: str | Path, value: object) -> Path:
    """Serialize JSON with stable formatting for reviews and diffs."""
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return write_text(path, encoded)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file through a context-managed text handle."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write CSV rows without leaving the handle open."""
    target = Path(path)
    names = list(fieldnames)
    if not names or any(not name for name in names):
        raise ValueError("fieldnames must contain non-empty names")

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return target


@contextmanager
def temporary_text_file(
    directory: str | Path,
    *,
    prefix: str = "lesson-",
) -> Iterator[Path]:
    """Yield a named temporary file and remove it on exit.

    The file is created in the caller's directory so a future atomic replace
    would remain on one filesystem. Cleanup still runs when the body raises.
    """
    parent = Path(directory)
    parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, raw_name = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(file_descriptor)
    temporary_path = Path(raw_name)
    try:
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def run_self_checks() -> None:
    """Check normal behavior, cleanup, and important edge cases."""
    with tempfile.TemporaryDirectory(prefix="python-course-day-7-") as raw_root:
        root = Path(raw_root)
        notes = root / "notes.txt"
        write_text(notes, " first line \n\nsecond line\n")
        assert read_text(notes) == " first line \n\nsecond line\n"
        assert read_nonempty_lines(notes) == ["first line", "second line"]

        numbers = root / "numbers.txt"
        write_text(numbers, "2\n\n4.5\n-1.5\n")
        assert summarize_numbers_file(numbers) == {
            "count": 3.0,
            "total": 5.0,
            "average": 1.67,
        }

        payload = root / "payload.json"
        write_json(payload, {"day": 7, "topics": ["paths", "contexts"]})
        assert json.loads(read_text(payload))["day"] == 7

        table = root / "people.csv"
        write_csv_rows(
            table,
            ["name", "score"],
            [{"name": "Asha", "score": 91}, {"name": "Bikash", "score": 84}],
        )
        assert read_csv_rows(table) == [
            {"name": "Asha", "score": "91"},
            {"name": "Bikash", "score": "84"},
        ]

        child = safe_child_path(root, "reports/summary.txt")
        assert child == root / "reports" / "summary.txt"
        try:
            safe_child_path(root, "../outside.txt")
        except ValueError:
            pass
        else:
            raise AssertionError("path traversal must be rejected")

        with temporary_text_file(root, prefix="check-") as temporary:
            assert temporary.exists()
            write_text(temporary, "temporary data")
        assert not temporary.exists()

        try:
            summarize_number_text("3\nnot-a-number\n")
        except ValueError as error:
            assert "line 2" in str(error)
        else:
            raise AssertionError("bad numeric lines must identify their line")

        for invalid_call in (
            lambda: write_text(root, ""),
            lambda: summarize_number_text(""),
            lambda: summarize_number_text("nan"),
            lambda: safe_child_path(root, "/etc/passwd"),
            lambda: write_csv_rows(root / "bad.csv", [], []),
        ):
            try:
                invalid_call()
            except (ValueError, OSError):
                pass
            else:
                raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without user input."""
    run_self_checks()

    with tempfile.TemporaryDirectory(prefix="python-course-day-7-demo-") as raw_root:
        root = Path(raw_root)
        log_path = root / "logs" / "events.txt"
        log_path.parent.mkdir()
        write_text(log_path, "started\nfinished\n")
        print("Non-empty log lines:", read_nonempty_lines(log_path))

        score_path = root / "scores.txt"
        write_text(score_path, "72\n88\n91\n")
        print("Score summary:", summarize_numbers_file(score_path))

        json_path = root / "settings.json"
        write_json(json_path, {"mode": "practice", "retries": 2})
        print("JSON keys:", sorted(json.loads(read_text(json_path))))

        csv_path = root / "scores.csv"
        write_csv_rows(
            csv_path,
            ["name", "score"],
            [{"name": "Asha", "score": 91}, {"name": "Bikash", "score": 84}],
        )
        print("CSV rows:", read_csv_rows(csv_path))
        print("Temporary workspace cleaned on exit.")

    print("Self-checks passed.")


if __name__ == "__main__":
    main()
