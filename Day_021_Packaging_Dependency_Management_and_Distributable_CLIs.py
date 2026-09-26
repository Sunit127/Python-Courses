"""Day 21: Packaging, dependency management, and distributable CLIs.

Learning goals
--------------
1. Distinguish a distribution name, an import package, and a command name.
2. Describe a modern project with pyproject.toml and a src layout.
3. Separate runtime dependencies from development tools.
4. Design a command-line interface around a testable application core.
5. Scaffold and inspect a package without publishing or installing anything.

Teaching notes
--------------
- A distribution is what an installer installs, an import package is what
  Python imports, and a console script is what a user types. Their names may
  differ: "course-report", "course_report", and "course-report" are valid
  names for those three roles.
- pyproject.toml is the source of build-system and project metadata. The
  [build-system] table selects a build backend; [project] records metadata,
  Python compatibility, dependencies, and console entry points.
- A src layout places import packages below src/. It prevents tests from
  accidentally importing the repository checkout instead of the installed
  package.
- Libraries should declare compatible ranges for direct runtime dependencies.
  Applications commonly lock exact transitive versions in a generated lock
  file. Do not hand-edit a generated lock file.
- Development tools such as test runners, type checkers, and builders are not
  runtime requirements. Optional dependency groups keep those concerns apart.
- A console script target names an importable no-argument function, for example
  course_report.cli:main. Return an integer status and raise SystemExit only at
  the outermost command boundary.
- Keep parsing and printing at the CLI boundary. Domain functions should accept
  Python values and return Python values so they can be tested without a shell.
- Build both a source distribution and a wheel, inspect their contents, install
  the wheel into a clean virtual environment, and run tests against that
  installation before publishing.
- Never upload credentials in configuration or source files. Trusted publishing
  or a narrowly scoped token is safer than embedding a long-lived password.

Run this file with Python 3.10+ to execute deterministic examples and checks.
It uses only the standard library and writes demonstrations to a temporary
directory that is removed automatically.

Practice exercises
------------------
1. Implement normalize_distribution_name(name) using the packaging name rule:
   trim the value, replace runs of hyphens, underscores, and dots with one
   hyphen, and compare names case-insensitively.
2. Implement split_dependency_groups(runtime, development). Validate every
   requirement, reject duplicates inside a group, and return immutable groups.
3. Implement render_pinned_requirements(requirements, resolved_versions).
   Produce deterministic name==version lines and fail when a direct dependency
   has no resolved version.

Solutions appear below the main examples.

Expert challenge: release-ready command package
-----------------------------------------------
Turn the generated course_report skeleton into a small production-quality
project. Add subcommands, configuration loading, structured errors, unit and
integration tests, static type checking, a changelog, and automated wheel
builds. Derive the installed version with importlib.metadata, not by importing
the build configuration.

Solution guidance
-----------------
1. Keep calculations in core.py and all terminal I/O in cli.py.
2. Test cli_main with explicit argument lists and captured output streams.
3. Build an sdist and wheel in an isolated environment, then inspect both.
4. Install the wheel into a new virtual environment and run its console script.
5. Check that only intended packages, type information, and documentation ship.
6. Tag releases immutably and publish from a protected CI environment.
7. Test invalid arguments, empty input, dependency conflicts, build failure,
   installation failure, and behavior of the installed artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


_NAME_SEPARATOR = re.compile(r"[-_.]+")
_DISTRIBUTION_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_IMPORT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
_PYTHON_REQUIREMENT = re.compile(r">=3\.(?:10|[1-9][0-9]+)")
_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:[<>=!~]=?[A-Za-z0-9.*+!-]+"
    r"(?:,[<>=!~]=?[A-Za-z0-9.*+!-]+)*)?"
)
_REQUIREMENT_NAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)")


def _clean_text(value: str, *, field: str) -> str:
    """Return stable non-empty text for project metadata."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field} cannot be empty")
    return cleaned


def normalize_distribution_name(name: str) -> str:
    """Solution 1: return a canonical, comparison-safe project name."""
    cleaned = _clean_text(name, field="distribution_name")
    normalized = _NAME_SEPARATOR.sub("-", cleaned).lower()
    if _DISTRIBUTION_NAME.fullmatch(normalized) is None:
        raise ValueError("distribution_name contains unsupported characters")
    return normalized


def _validate_import_name(name: str) -> str:
    cleaned = _clean_text(name, field="import_name")
    if _IMPORT_NAME.fullmatch(cleaned) is None:
        raise ValueError("import_name must be one valid Python identifier")
    if cleaned in {"False", "None", "True"}:
        raise ValueError("import_name cannot be a reserved constant")
    return cleaned


def _validate_version(value: str) -> str:
    cleaned = _clean_text(value, field="version")
    if _VERSION.fullmatch(cleaned) is None:
        raise ValueError("version must use MAJOR.MINOR.PATCH integers")
    return cleaned


def _validate_python_requirement(value: str) -> str:
    cleaned = _clean_text(value, field="requires_python")
    if _PYTHON_REQUIREMENT.fullmatch(cleaned) is None:
        raise ValueError("requires_python must look like >=3.10")
    return cleaned


def _validate_requirement(value: str) -> str:
    """Validate the deliberately small requirement subset used in this lesson."""
    cleaned = _clean_text(value, field="requirement")
    if _REQUIREMENT.fullmatch(cleaned) is None:
        raise ValueError(
            "requirement must use a project name and optional version specifiers"
        )
    return cleaned


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(_validate_requirement(requirement))
    if match is None:
        raise ValueError("requirement has no project name")
    return normalize_distribution_name(match.group(1))


def _validated_group(requirements: Iterable[str]) -> tuple[str, ...]:
    materialized = tuple(_validate_requirement(item) for item in requirements)
    names = tuple(_requirement_name(item) for item in materialized)
    if len(set(names)) != len(names):
        raise ValueError("a dependency group cannot contain duplicate projects")
    return materialized


def split_dependency_groups(
    runtime: Iterable[str],
    development: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Solution 2: validate and freeze runtime and development dependencies."""
    return _validated_group(runtime), _validated_group(development)


@dataclass(frozen=True, slots=True)
class ProjectSpec:
    """Validated metadata used by the rendering and scaffolding examples."""

    distribution_name: str
    import_name: str
    version: str
    description: str
    requires_python: str = ">=3.10"
    dependencies: tuple[str, ...] = ()
    development_dependencies: tuple[str, ...] = (
        "build>=1.2,<2",
        "pytest>=8,<9",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distribution_name",
            normalize_distribution_name(self.distribution_name),
        )
        object.__setattr__(
            self,
            "import_name",
            _validate_import_name(self.import_name),
        )
        object.__setattr__(self, "version", _validate_version(self.version))
        object.__setattr__(
            self,
            "description",
            _clean_text(self.description, field="description"),
        )
        object.__setattr__(
            self,
            "requires_python",
            _validate_python_requirement(self.requires_python),
        )
        runtime, development = split_dependency_groups(
            self.dependencies,
            self.development_dependencies,
        )
        object.__setattr__(self, "dependencies", runtime)
        object.__setattr__(self, "development_dependencies", development)


def _toml_string(value: str) -> str:
    """Encode a basic TOML string with JSON's compatible escaping rules."""
    return json.dumps(value, ensure_ascii=True)


def _toml_array(values: Iterable[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def render_pyproject(spec: ProjectSpec) -> str:
    """Render readable PEP 621 metadata for a Hatchling-based package."""
    if not isinstance(spec, ProjectSpec):
        raise TypeError("spec must be a ProjectSpec")

    command_name = spec.distribution_name
    package_path = f"src/{spec.import_name}"
    lines = [
        "[build-system]",
        'requires = ["hatchling>=1.26,<2"]',
        'build-backend = "hatchling.build"',
        "",
        "[project]",
        f"name = {_toml_string(spec.distribution_name)}",
        f"version = {_toml_string(spec.version)}",
        f"description = {_toml_string(spec.description)}",
        f"requires-python = {_toml_string(spec.requires_python)}",
        f"dependencies = {_toml_array(spec.dependencies)}",
        "",
        "[project.optional-dependencies]",
        f"dev = {_toml_array(spec.development_dependencies)}",
        "",
        "[project.scripts]",
        (
            f"{_toml_string(command_name)} = "
            f"{_toml_string(spec.import_name + '.cli:main')}"
        ),
        "",
        "[tool.hatch.build.targets.wheel]",
        f"packages = [{_toml_string(package_path)}]",
        "",
    ]
    return "\n".join(lines)


def project_layout(spec: ProjectSpec) -> tuple[str, ...]:
    """Return the intended files without touching the filesystem."""
    if not isinstance(spec, ProjectSpec):
        raise TypeError("spec must be a ProjectSpec")
    package = f"src/{spec.import_name}"
    return (
        "README.md",
        "pyproject.toml",
        f"{package}/__init__.py",
        f"{package}/__main__.py",
        f"{package}/cli.py",
        f"{package}/core.py",
        "tests/test_core.py",
    )


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Small domain result used by the CLI example."""

    count: int
    minimum: float
    maximum: float
    average: float


def summarize_scores(scores: Iterable[float]) -> ScoreSummary:
    """Calculate a summary after rejecting Booleans and non-finite values."""
    values: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("scores must be numbers")
        number = float(score)
        if not math.isfinite(number):
            raise ValueError("scores must be finite")
        values.append(number)
    if not values:
        raise ValueError("provide at least one score")
    return ScoreSummary(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        average=round(sum(values) / len(values), 2),
    )


def parse_scores(values: Iterable[str]) -> tuple[float, ...]:
    """Convert CLI text at the boundary and report the offending value."""
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"not a number: {value!r}") from error
        if not math.isfinite(number):
            raise ValueError(f"score must be finite: {value!r}")
        parsed.append(number)
    return tuple(parsed)


class CliUsageError(ValueError):
    """An expected command-line usage failure."""


class CourseArgumentParser(argparse.ArgumentParser):
    """Raise a testable exception instead of terminating deep in the parser."""

    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser() -> CourseArgumentParser:
    """Create the command parser without reading process-global arguments."""
    parser = CourseArgumentParser(prog="course-report")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser(
        "summary",
        help="summarize one or more numeric scores",
    )
    summary_parser.add_argument("scores", nargs="+")

    subparsers.add_parser("layout", help="show the recommended package layout")
    subparsers.add_parser("pyproject", help="show example project metadata")
    return parser


def run_cli(
    argv: Sequence[str],
    *,
    spec: ProjectSpec,
) -> tuple[int, str]:
    """Run CLI application logic without printing or raising SystemExit."""
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of argument strings")
    namespace = build_parser().parse_args(list(argv))

    if namespace.command == "summary":
        summary = summarize_scores(parse_scores(namespace.scores))
        output = (
            f"count={summary.count} minimum={summary.minimum:g} "
            f"maximum={summary.maximum:g} average={summary.average:g}"
        )
        return 0, output
    if namespace.command == "layout":
        return 0, "\n".join(project_layout(spec))
    if namespace.command == "pyproject":
        return 0, render_pyproject(spec).rstrip()
    raise AssertionError(f"unhandled command: {namespace.command!r}")


def cli_main(
    argv: Sequence[str] | None = None,
    *,
    spec: ProjectSpec | None = None,
) -> int:
    """Console boundary: print results and translate expected errors."""
    arguments = sys.argv[1:] if argv is None else argv
    project = example_project() if spec is None else spec
    try:
        status, output = run_cli(arguments, spec=project)
    except (CliUsageError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return status


def installed_version(distribution_name: str, *, fallback: str) -> str:
    """Read installed metadata without importing the distribution."""
    normalized = normalize_distribution_name(distribution_name)
    validated_fallback = _validate_version(fallback)
    try:
        return package_version(normalized)
    except PackageNotFoundError:
        return validated_fallback


def render_pinned_requirements(
    requirements: Iterable[str],
    resolved_versions: Mapping[str, str],
) -> str:
    """Solution 3: render deterministic pins from a resolver's result."""
    group = _validated_group(requirements)
    normalized_versions = {
        normalize_distribution_name(name): _validate_version(value)
        for name, value in resolved_versions.items()
    }

    pins: list[str] = []
    for requirement in group:
        name = _requirement_name(requirement)
        try:
            resolved = normalized_versions[name]
        except KeyError as error:
            raise LookupError(f"no resolved version for {name!r}") from error
        pins.append(f"{name}=={resolved}")
    return "\n".join(sorted(pins)) + ("\n" if pins else "")


def scaffold_project(root: str | Path, spec: ProjectSpec) -> tuple[Path, ...]:
    """Create a minimal package only inside an empty caller-selected directory."""
    if not isinstance(spec, ProjectSpec):
        raise TypeError("spec must be a ProjectSpec")
    destination = Path(root)
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(destination)
        if any(destination.iterdir()):
            raise FileExistsError("destination must be empty")
    else:
        destination.mkdir(parents=True)

    package = destination / "src" / spec.import_name
    tests = destination / "tests"
    package.mkdir(parents=True)
    tests.mkdir()

    files: dict[Path, str] = {
        destination / "pyproject.toml": render_pyproject(spec),
        destination / "README.md": (
            f"# {spec.distribution_name}\n\n{spec.description}\n"
        ),
        package / "__init__.py": (
            '"""Public package metadata."""\n\n'
            f'__version__ = "{spec.version}"\n'
        ),
        package / "__main__.py": (
            "from .cli import main\n\n"
            'if __name__ == "__main__":\n'
            "    raise SystemExit(main())\n"
        ),
        package / "core.py": (
            '"""Pure application calculations."""\n\n'
            "def mean(values: list[float]) -> float:\n"
            "    if not values:\n"
            '        raise ValueError("provide at least one value")\n'
            "    return sum(values) / len(values)\n"
        ),
        package / "cli.py": (
            '"""Console entry point."""\n\n'
            "import argparse\n"
            "from .core import mean\n\n"
            "def main() -> int:\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("values", nargs="+", type=float)\n'
            "    arguments = parser.parse_args()\n"
            '    print(f"{mean(arguments.values):g}")\n'
            "    return 0\n"
        ),
        tests / "test_core.py": (
            f"from {spec.import_name}.core import mean\n\n"
            "def test_mean() -> None:\n"
            "    assert mean([2.0, 4.0]) == 3.0\n"
        ),
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8", newline="")
    return tuple(sorted(files, key=lambda path: path.as_posix()))


def example_project() -> ProjectSpec:
    """Return one reusable example with no runtime dependencies."""
    return ProjectSpec(
        distribution_name="course-report",
        import_name="course_report",
        version="1.0.0",
        description="A small, testable score reporting command.",
    )


def run_self_checks() -> None:
    """Verify metadata, CLI behavior, scaffolding, and edge cases."""
    spec = example_project()
    assert normalize_distribution_name(" Course.Report_tools ") == (
        "course-report-tools"
    )
    assert spec.distribution_name == "course-report"
    assert spec.import_name == "course_report"
    assert project_layout(spec)[2] == "src/course_report/__init__.py"

    metadata = render_pyproject(spec)
    assert '[project]' in metadata
    assert 'name = "course-report"' in metadata
    assert '"course-report" = "course_report.cli:main"' in metadata
    assert 'packages = ["src/course_report"]' in metadata
    assert metadata.endswith("\n")

    status, summary = run_cli(["summary", "10", "20", "30"], spec=spec)
    assert status == 0
    assert summary == "count=3 minimum=10 maximum=30 average=20"
    assert run_cli(["layout"], spec=spec)[1].splitlines() == list(
        project_layout(spec)
    )

    assert render_pinned_requirements(
        ("httpx>=0.27,<1", "rich>=13,<14"),
        {"rich": "13.9.4", "HTTPX": "0.28.1"},
    ) == "httpx==0.28.1\nrich==13.9.4\n"

    with TemporaryDirectory(prefix="python-course-day-21-") as raw_directory:
        root = Path(raw_directory)
        created = scaffold_project(root, spec)
        relative = tuple(path.relative_to(root).as_posix() for path in created)
        assert set(relative) == set(project_layout(spec))
        assert (root / "src" / "course_report" / "cli.py").is_file()
        assert render_pyproject(spec) == (root / "pyproject.toml").read_text(
            encoding="utf-8"
        )

    invalid_calls = (
        lambda: normalize_distribution_name("---"),
        lambda: ProjectSpec("demo", "bad-name", "1.0.0", "description"),
        lambda: ProjectSpec("demo", "demo", "1.0", "description"),
        lambda: split_dependency_groups(("HTTPX>=1", "httpx<2"), ()),
        lambda: parse_scores(("nan",)),
        lambda: summarize_scores(()),
        lambda: run_cli(["unknown"], spec=spec),
        lambda: render_pinned_requirements(("httpx>=1",), {}),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (CliUsageError, LookupError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run deterministic demonstrations without installing or publishing."""
    run_self_checks()
    spec = example_project()

    print("Recommended layout:")
    print(run_cli(["layout"], spec=spec)[1])
    print("\nScore command:")
    print(run_cli(["summary", "72", "88", "91"], spec=spec)[1])
    print("\nGenerated project metadata:")
    print(render_pyproject(spec).rstrip())
    print("\nSelf-checks passed.")


if __name__ == "__main__":
    main()
