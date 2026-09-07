"""Day 6: Modules, imports, and package structure.

Learning goals
--------------
1. Explain why modules give code a clear namespace and reusable boundary.
2. Use import styles deliberately and avoid import-time side effects.
3. Expose a small public API while keeping implementation details private.
4. Organize related modules into a package with absolute imports.
5. Build an explicit command registry instead of unsafe dynamic execution.

Teaching notes
--------------
- Every .py file is a module. A package groups related modules in a directory;
  __init__.py can define the package's public API.
- "import statistics as stats" keeps the module namespace visible.
  "from pathlib import PurePath" is useful for a small, frequently used API.
  Avoid star imports because readers cannot see where names began.
- Imports run a module's top-level statements once per Python process, then
  cache the module in sys.modules. Keep top-level work cheap and deterministic:
  define objects there, but put demonstrations, network calls, and command-line
  input behind a main function.
- The __name__ guard runs the demo only when this file is executed, not when
  another module imports it.
- A leading underscore marks an internal name by convention. __all__ makes an
  intended public API explicit for documentation and star imports, although
  ordinary attribute access is still possible.
- Prefer absolute package imports such as "from course_tools.commands import
  total". Relative imports are useful inside a tightly coupled package but
  become difficult to follow when deeply nested.
- Import only trusted, expected modules. Never turn unchecked user text into an
  import name or pass it to eval or exec.

A small package could grow from this lesson like this:

    course_tools/
        __init__.py          # re-export the supported public API
        calculations.py     # score summaries and numeric helpers
        commands.py         # command functions and the registry
        cli.py              # input/output and main()
    tests/
        test_calculations.py
        test_commands.py

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Write module_name_from_path(path) to convert a relative Python file path
   such as app/services/pricing.py into app.services.pricing. Treat
   app/__init__.py as app and reject invalid paths.
2. Write merge_exports(*groups) to combine public names without duplicates,
   preserving their first-seen order.
3. Add a "maximum" command to COMMANDS without changing run_command.

Solutions and clear guidance appear below the examples.

Expert challenge: extensible command package
--------------------------------------------
Split the command runner in this file into the package layout shown above.
Keep each command as a function with the same signature, expose only supported
commands from the package, and make the CLI depend on the registry rather than
a chain of if/elif statements. Add tests for an unknown command, empty input,
non-finite numbers, and a successful command. For an advanced extension, load
plugins only from a hard-coded allowlist or installed entry points; do not
import arbitrary names supplied at the command line.
"""

import importlib
import math
import statistics as stats
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import PurePath, PurePosixPath

__all__ = [
    "COMMANDS",
    "available_public_names",
    "count_file_types",
    "load_public_callable",
    "merge_exports",
    "module_name_from_path",
    "run_command",
    "summarize_scores",
]

Command = Callable[[list[str]], str]
_ALLOWED_DEMO_MODULES = frozenset({"math", "statistics"})


def summarize_scores(scores: Iterable[float]) -> dict[str, float]:
    """Return common summary statistics for finite numeric scores."""
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("provide at least one score")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")

    return {
        "count": float(len(values)),
        "mean": round(stats.fmean(values), 2),
        "median": round(stats.median(values), 2),
        "minimum": min(values),
        "maximum": max(values),
    }


def count_file_types(paths: Iterable[str]) -> dict[str, int]:
    """Count case-insensitive filename extensions without touching the disk."""
    counts: Counter[str] = Counter()
    for path_text in paths:
        cleaned = path_text.strip()
        if not cleaned:
            raise ValueError("paths cannot be empty")
        suffix = PurePath(cleaned).suffix.casefold()
        counts[suffix or "[no extension]"] += 1
    return dict(counts)


def available_public_names(namespace: Mapping[str, object]) -> list[str]:
    """Return sorted public identifiers from a module-like namespace."""
    return sorted(
        name
        for name in namespace
        if name.isidentifier() and not name.startswith("_")
    )


def load_public_callable(
    module_name: str,
    attribute_name: str,
    *,
    allowed_modules: frozenset[str] = _ALLOWED_DEMO_MODULES,
) -> Callable[..., object]:
    """Load a public callable from an explicit module allowlist.

    Dynamic imports are sometimes appropriate for plugin systems. The
    allowlist and public-name check keep this teaching example predictable.
    """
    if module_name not in allowed_modules:
        raise PermissionError(f"module {module_name!r} is not allowed")
    if not attribute_name.isidentifier() or attribute_name.startswith("_"):
        raise ValueError("attribute must be a public Python identifier")

    module = importlib.import_module(module_name)
    try:
        candidate = getattr(module, attribute_name)
    except AttributeError as error:
        raise LookupError(
            f"{module_name!r} has no public attribute {attribute_name!r}"
        ) from error

    if not callable(candidate):
        raise TypeError(f"{module_name}.{attribute_name} is not callable")
    return candidate


# Practice exercise solutions


def module_name_from_path(path: str) -> str:
    """Solution 1: convert a relative .py path into an importable module name."""
    normalized = path.strip().replace("\\", "/")
    module_path = PurePosixPath(normalized)
    if not normalized or module_path.is_absolute() or module_path.suffix != ".py":
        raise ValueError("path must be a relative .py file")
    if any(part in {".", ".."} for part in module_path.parts):
        raise ValueError("path cannot contain current or parent directory parts")

    parts = list(module_path.parts)
    filename = parts.pop()
    stem = PurePosixPath(filename).stem
    if stem != "__init__":
        parts.append(stem)
    if not parts or any(not part.isidentifier() for part in parts):
        raise ValueError("every module path component must be an identifier")
    return ".".join(parts)


def merge_exports(*groups: Iterable[str]) -> list[str]:
    """Solution 2: merge valid public names in first-seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            if not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"{name!r} is not a public Python identifier")
            if name not in seen:
                seen.add(name)
                merged.append(name)
    return merged


def _parse_finite_numbers(arguments: list[str]) -> list[float]:
    """Parse one or more finite numbers for numeric commands."""
    if not arguments:
        raise ValueError("provide at least one number")

    try:
        numbers = [float(argument) for argument in arguments]
    except ValueError as error:
        raise ValueError("all arguments must be numbers") from error

    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("numbers must be finite")
    return numbers


def _total_command(arguments: list[str]) -> str:
    """Return the sum of numeric command arguments."""
    result = sum(_parse_finite_numbers(arguments))
    if not math.isfinite(result):
        raise ValueError("total is outside the finite float range")
    return f"{result:g}"


def _mean_command(arguments: list[str]) -> str:
    """Return the arithmetic mean of numeric command arguments."""
    result = stats.fmean(_parse_finite_numbers(arguments))
    if not math.isfinite(result):
        raise ValueError("mean is outside the finite float range")
    return f"{result:g}"


def _unique_command(arguments: list[str]) -> str:
    """Return comma-separated unique values in their original order."""
    cleaned = [argument.strip() for argument in arguments]
    if not cleaned or any(not argument for argument in cleaned):
        raise ValueError("provide one or more non-empty values")
    return ", ".join(dict.fromkeys(cleaned))


COMMANDS: dict[str, Command] = {
    "mean": _mean_command,
    "total": _total_command,
    "unique": _unique_command,
}


def run_command(
    command_name: str,
    arguments: Iterable[str],
    registry: Mapping[str, Command] = COMMANDS,
) -> str:
    """Run one explicitly registered command with materialized arguments."""
    cleaned_name = command_name.strip().casefold()
    try:
        command = registry[cleaned_name]
    except KeyError as error:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            f"unknown command {command_name!r}; choose from: {choices}"
        ) from error
    return command(list(arguments))


def run_self_checks() -> None:
    """Check imports, examples, exercise solutions, and important edge cases."""
    assert summarize_scores([80, 90, 100]) == {
        "count": 3.0,
        "mean": 90.0,
        "median": 90.0,
        "minimum": 80.0,
        "maximum": 100.0,
    }
    assert count_file_types(["report.CSV", "notes.csv", "README", ".env"]) == {
        ".csv": 2,
        "[no extension]": 2,
    }
    assert available_public_names(
        {"visible": object(), "_internal": object(), "not-valid": object()}
    ) == ["visible"]

    square_root = load_public_callable("math", "sqrt")
    assert square_root(81) == 9
    assert module_name_from_path("app/services/pricing.py") == (
        "app.services.pricing"
    )
    assert module_name_from_path("app/__init__.py") == "app"
    assert merge_exports(["total", "mean"], ["mean", "unique"]) == [
        "total",
        "mean",
        "unique",
    ]
    assert run_command(" TOTAL ", ["2", "3.5"]) == "5.5"
    assert run_command("mean", ["2", "4", "6"]) == "4"
    assert run_command("unique", ["red", "blue", "red"]) == "red, blue"

    invalid_calls = (
        lambda: summarize_scores([]),
        lambda: summarize_scores([float("nan")]),
        lambda: count_file_types([""]),
        lambda: load_public_callable("os", "getcwd"),
        lambda: load_public_callable("math", "_secret"),
        lambda: module_name_from_path("../unsafe.py"),
        lambda: module_name_from_path("notes.txt"),
        lambda: merge_exports(["public", "_private"]),
        lambda: run_command("missing", []),
        lambda: run_command("total", ["inf"]),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (LookupError, PermissionError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without user input."""
    run_self_checks()

    print("Score summary:", summarize_scores([72, 88, 91, 88]))
    print(
        "File types:",
        count_file_types(["lesson.py", "README.md", "scores.CSV", "LICENSE"]),
    )
    print("Module path:", module_name_from_path("course_tools/commands.py"))
    print("Public exports:", merge_exports(["summarize"], ["run", "summarize"]))
    print("Square root:", load_public_callable("math", "sqrt")(144))
    print("Command result:", run_command("mean", ["10", "20", "30"]))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
