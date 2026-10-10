"""Day 34: Continuous integration, quality gates, and reproducible releases.

Learning goals
--------------
1. Model CI as ordered, deterministic checks with required and advisory outcomes.
2. Keep quality gates explicit so a failed required check blocks a release.
3. Validate Python source safely without relying on the current working directory.
4. Build canonical artifact manifests with stable paths, sizes, and SHA-256 digests.
5. Create release metadata that can be audited and reproduced from a revision.

Teaching notes
--------------
- Continuous integration is a feedback loop, not merely a server that runs tests.
  A useful pipeline checks formatting, syntax, types, tests, documentation, and
  packaging in a deliberate order.
- A quality gate should be fail-closed for required checks. Advisory checks can
  report warnings, but they must never make a red result look green.
- Make checks small and injectable. The orchestration layer should not know how a
  formatter, type checker, or test runner is implemented. In a real repository,
  adapters may invoke tools with subprocess, but this lesson keeps examples pure.
- Reproducibility means the same source revision and declared inputs produce the
  same artifact bytes or the same digest. Sort paths, normalize metadata, pin
  dependencies, and inject the build timestamp instead of reading the clock.
- A manifest is evidence, not a security boundary. Sign release metadata, protect
  the CI credentials, and publish provenance through a trusted build system.
- Never include tokens, full environment dumps, or untrusted command output in a
  release record. Logs should be bounded and redacted.
- CI should run on a clean checkout and enforce the same commands locally and in
  automation. A green local run is not proof that the pipeline is reproducible.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, and writes no files.

Practice exercises
------------------
1. Implement parse_version(value) for MAJOR.MINOR.PATCH versions with an optional
   prerelease suffix. Reject whitespace, negative numbers, and empty suffixes.
2. Implement required_failures(results). Return the failed required check names in
   execution order, without exposing mutable internal collections.
3. Implement manifest_for_files(files). Return sorted immutable artifact entries
   with normalized relative paths and SHA-256 digests.

Solutions appear below the main example.

Expert challenge: a release pipeline
-------------------------------------
Design a package that runs in a clean checkout and publishes a wheel plus a
source archive. Add a locked dependency input, a test matrix for supported Python
versions, a strict type-checking job, a vulnerability scan, and a signed
provenance statement. Store the exact revision, build tool versions, normalized
artifact manifest, and check results. Upload artifacts only after all required
checks pass.

Solution guidance
-----------------
1. Give every check a stable name, an explicit required flag, and a bounded
   output. Execute checks in a fixed order and preserve each result.
2. Keep the release job separate from pull-request checks. Rebuild from the
   protected revision rather than trusting an artifact uploaded by a contributor.
3. Normalize every archive path to POSIX form, reject absolute paths and parent
   traversal, and sort entries before hashing.
4. Pin build dependencies and use a fixed source date epoch or injected timestamp.
   Exclude host-specific paths, usernames, and nondeterministic file ordering.
5. Sign the final manifest with a key held outside the build worker. Verify the
   signature and the artifact digest after download, then retain the evidence.
6. Test missing checks, duplicate names, malformed versions, changed bytes, failed
   required checks, advisory failures, and two runs with identical inputs.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re


class QualityError(Exception):
    """Base class for expected quality-pipeline failures."""


class ValidationError(QualityError, ValueError):
    """An input does not satisfy the pipeline contract."""


class QualityGateFailed(QualityError):
    """At least one required check failed."""


class ReproducibilityError(QualityError):
    """A release manifest no longer matches its declared inputs."""


_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z.-]+))?$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")


def _clean_name(value: str, *, field: str) -> str:
    """Validate a short machine-readable name."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 80:
        raise ValidationError(f"{field} must be 1 to 80 characters")
    if not re.fullmatch(r"[a-z][a-z0-9._-]*", cleaned):
        raise ValidationError(f"{field} contains unsupported characters")
    return cleaned


def _clean_revision(value: str) -> str:
    """Validate a Git-like immutable revision identifier."""
    if not isinstance(value, str) or not _REVISION_PATTERN.fullmatch(value):
        raise ValidationError("revision must be a 40-character lowercase hex SHA")
    return value


def _canonical_json(value: object, *, field: str) -> str:
    """Encode JSON deterministically and reject non-finite numbers."""
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field} is not canonical JSON data") from error


def _sha256(data: bytes) -> str:
    """Hash bytes using the digest recorded in manifests."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def normalize_artifact_path(path: str) -> str:
    """Return a safe POSIX relative path for an artifact entry."""
    if not isinstance(path, str):
        raise TypeError("artifact path must be text")
    candidate = path.replace("\\", "/")
    if (
        not candidate
        or candidate.startswith("/")
        or candidate.startswith("./")
        or candidate.endswith("/")
        or "//" in candidate
        or not _PATH_PATTERN.fullmatch(candidate)
    ):
        raise ValidationError(f"invalid artifact path: {path!r}")
    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValidationError(f"artifact path escapes its root: {path!r}")
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class Version:
    """A validated semantic version used by a release manifest."""

    major: int
    minor: int
    patch: int
    prerelease: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{field_name} must be a non-negative integer")
        if self.prerelease is not None:
            if (
                not isinstance(self.prerelease, str)
                or not self.prerelease
                or not re.fullmatch(r"[0-9A-Za-z.-]+", self.prerelease)
            ):
                raise ValidationError("prerelease must contain non-empty ASCII labels")

    def __str__(self) -> str:
        suffix = "" if self.prerelease is None else f"-{self.prerelease}"
        return f"{self.major}.{self.minor}.{self.patch}{suffix}"


def parse_version(value: str) -> Version:
    """Solution 1: parse a strict MAJOR.MINOR.PATCH version."""
    if not isinstance(value, str) or value != value.strip():
        raise ValidationError("version must not have surrounding whitespace")
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValidationError("version must be MAJOR.MINOR.PATCH with an optional suffix")
    major, minor, patch = (int(match.group(index)) for index in range(1, 4))
    return Version(major, minor, patch, match.group(4))


@dataclass(frozen=True, slots=True)
class Check:
    """One deterministic quality check and its severity."""

    name: str
    required: bool
    action: Callable[[], str | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field="check name"))
        if not isinstance(self.required, bool):
            raise TypeError("required must be a Boolean")
        if not callable(self.action):
            raise TypeError("check action must be callable")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """An immutable outcome safe to serialize into CI evidence."""

    name: str
    required: bool
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field="result name"))
        if not isinstance(self.required, bool) or not isinstance(self.passed, bool):
            raise TypeError("required and passed must be Booleans")
        if not isinstance(self.detail, str) or len(self.detail) > 500:
            raise ValidationError("check detail must be short text")


@dataclass(frozen=True, slots=True)
class GateReport:
    """Ordered results from one quality-gate invocation."""

    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every required check passed."""
        return all(result.passed for result in self.results if result.required)

    @property
    def failed_required(self) -> tuple[str, ...]:
        """Return failed required names without exposing mutable state."""
        return tuple(
            result.name
            for result in self.results
            if result.required and not result.passed
        )

    @property
    def failed_advisory(self) -> tuple[str, ...]:
        """Return failed advisory names in execution order."""
        return tuple(
            result.name
            for result in self.results
            if not result.required and not result.passed
        )


def run_check(check: Check) -> CheckResult:
    """Run one check and convert expected failures into bounded evidence."""
    try:
        detail = check.action()
    except Exception as error:  # CI must report a check failure, not hide it.
        message = str(error).replace("\n", " ").strip()
        safe_detail = f"{type(error).__name__}: {message}"[:500]
        return CheckResult(check.name, check.required, False, safe_detail)
    if detail is not None and not isinstance(detail, str):
        raise TypeError("a check must return text or None")
    return CheckResult(check.name, check.required, True, detail or "passed")


def run_quality_gate(checks: Iterable[Check]) -> GateReport:
    """Run checks in order and reject duplicate or missing names."""
    materialized = tuple(checks)
    if not materialized:
        raise ValidationError("a quality gate needs at least one check")
    names = tuple(check.name for check in materialized)
    if len(set(names)) != len(names):
        raise ValidationError("check names must be unique")
    return GateReport(tuple(run_check(check) for check in materialized))


def require_quality_gate(report: GateReport) -> None:
    """Raise a release-blocking error when required checks fail."""
    if not isinstance(report, GateReport):
        raise TypeError("report must be a GateReport")
    if report.failed_required:
        joined = ", ".join(report.failed_required)
        raise QualityGateFailed(f"required checks failed: {joined}")


def required_failures(results: Iterable[CheckResult]) -> tuple[str, ...]:
    """Solution 2: return failed required names in execution order."""
    materialized = tuple(results)
    if any(not isinstance(result, CheckResult) for result in materialized):
        raise TypeError("results must contain CheckResult values")
    return tuple(
        result.name
        for result in materialized
        if result.required and not result.passed
    )


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """A normalized artifact path and its content identity."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_artifact_path(self.path))
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValidationError("artifact size must be a non-negative integer")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValidationError("artifact sha256 must be lowercase hexadecimal")


def manifest_for_files(files: Mapping[str, bytes]) -> tuple[ArtifactEntry, ...]:
    """Solution 3: normalize, sort, and hash artifact files."""
    if not isinstance(files, Mapping):
        raise TypeError("files must be a mapping")
    entries: list[ArtifactEntry] = []
    seen: set[str] = set()
    for raw_path, data in files.items():
        path = normalize_artifact_path(raw_path)
        if path in seen:
            raise ValidationError(f"duplicate artifact path: {path}")
        if not isinstance(data, bytes):
            raise TypeError(f"artifact {path!r} must contain bytes")
        seen.add(path)
        entries.append(ArtifactEntry(path, len(data), _sha256(data)))
    return tuple(sorted(entries, key=lambda entry: entry.path))


def manifest_object(entries: Iterable[ArtifactEntry]) -> list[dict[str, object]]:
    """Convert entries to fresh JSON-compatible dictionaries."""
    materialized = tuple(entries)
    if tuple(entry.path for entry in materialized) != tuple(
        sorted(entry.path for entry in materialized)
    ):
        raise ValidationError("manifest entries must be sorted by path")
    return [
        {"path": entry.path, "size": entry.size, "sha256": entry.sha256}
        for entry in materialized
    ]


def build_release_manifest(
    project: str,
    version: Version,
    revision: str,
    files: Mapping[str, bytes],
    *,
    built_at: datetime,
) -> str:
    """Create canonical release metadata from explicitly supplied inputs."""
    project = _clean_name(project, field="project")
    if not isinstance(version, Version):
        raise TypeError("version must be a Version")
    revision = _clean_revision(revision)
    if not isinstance(built_at, datetime) or built_at.tzinfo is None:
        raise ValidationError("built_at must be timezone-aware")
    normalized_time = built_at.astimezone(timezone.utc).replace(microsecond=0)
    entries = manifest_for_files(files)
    payload = {
        "project": project,
        "version": str(version),
        "revision": revision,
        "built_at": normalized_time.isoformat().replace("+00:00", "Z"),
        "artifacts": manifest_object(entries),
    }
    return _canonical_json(payload, field="release manifest")


def verify_release_manifest(
    manifest_text: str,
    files: Mapping[str, bytes],
) -> None:
    """Verify that current bytes match every declared release artifact."""
    if not isinstance(manifest_text, str):
        raise TypeError("manifest_text must be text")
    try:
        payload = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ReproducibilityError("manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ReproducibilityError("manifest must be an object")
    try:
        declared = tuple(
            ArtifactEntry(
                item["path"],
                item["size"],
                item["sha256"],
            )
            for item in payload["artifacts"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReproducibilityError("manifest artifacts are malformed") from error
    actual = manifest_for_files(files)
    if declared != actual:
        raise ReproducibilityError("artifact bytes do not match the manifest")


def syntax_check(sources: Mapping[str, str]) -> str:
    """Compile source text without importing or executing it."""
    if not isinstance(sources, Mapping) or not sources:
        raise ValidationError("sources must be a non-empty mapping")
    count = 0
    for raw_path, source in sorted(sources.items()):
        path = normalize_artifact_path(raw_path)
        if not path.endswith(".py"):
            raise ValidationError(f"syntax check received non-Python path: {path}")
        if not isinstance(source, str):
            raise TypeError(f"source for {path!r} must be text")
        compile(source, path, "exec", dont_inherit=True)
        count += 1
    return f"{count} Python source file(s) compile"


def documentation_check(readme: str, required_sections: Iterable[str]) -> str:
    """Check required documentation markers without accepting empty headings."""
    if not isinstance(readme, str):
        raise TypeError("readme must be text")
    sections = tuple(required_sections)
    if not sections or any(not isinstance(section, str) or not section.strip() for section in sections):
        raise ValidationError("required_sections must contain non-empty text")
    missing = tuple(section for section in sections if section not in readme)
    if missing:
        raise ValidationError("missing documentation: " + ", ".join(missing))
    return f"{len(sections)} documentation section(s) present"


def run_self_checks() -> None:
    """Exercise passing, advisory, blocking, and reproducibility paths."""
    sources = {
        "src/course/core.py": "def add(left: int, right: int) -> int:\n    return left + right\n",
        "tests/test_core.py": "assert 2 + 2 == 4\n",
    }
    files = {
        "README.md": b"# Course\n",
        "src/course/core.py": sources["src/course/core.py"].encode("utf-8"),
    }

    report = run_quality_gate(
        (
            Check("syntax", True, lambda: syntax_check(sources)),
            Check("tests", True, lambda: "2 tests passed"),
            Check("typing", False, lambda: (_ for _ in ()).throw(
                ValidationError("third-party checker unavailable in this demo")
            )),
            Check("docs", True, lambda: documentation_check(
                "# Course\n## Roadmap\n## Release process\n",
                ("Roadmap", "Release process"),
            )),
        )
    )
    assert report.passed
    assert report.failed_required == ()
    assert report.failed_advisory == ("typing",)
    assert required_failures(report.results) == ()
    require_quality_gate(report)

    blocked = run_quality_gate(
        (
            Check("syntax", True, lambda: "passed"),
            Check("tests", True, lambda: (_ for _ in ()).throw(
                AssertionError("one test failed")
            )),
        )
    )
    assert blocked.passed is False
    assert required_failures(blocked.results) == ("tests",)
    try:
        require_quality_gate(blocked)
    except QualityGateFailed:
        pass
    else:
        raise AssertionError("a failed required check must block release")

    version = parse_version("2.4.0-rc.1")
    assert str(version) == "2.4.0-rc.1"
    assert parse_version("0.0.1").patch == 1
    assert manifest_for_files(files)[0].path == "README.md"

    timestamp = datetime(2026, 10, 10, 9, 0, 1, tzinfo=timezone.utc)
    manifest = build_release_manifest(
        "python-course",
        version,
        "0123456789abcdef0123456789abcdef01234567",
        files,
        built_at=timestamp,
    )
    verify_release_manifest(manifest, files)
    assert '"built_at":"2026-10-10T09:00:01Z"' in manifest

    changed_files = {**files, "README.md": b"# Changed\n"}
    try:
        verify_release_manifest(manifest, changed_files)
    except ReproducibilityError:
        pass
    else:
        raise AssertionError("changed bytes must invalidate a manifest")

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: parse_version("1.2"),
        lambda: parse_version(" 1.2.3"),
        lambda: normalize_artifact_path("../secret.txt"),
        lambda: normalize_artifact_path("/absolute.txt"),
        lambda: manifest_for_files({"README.md": "text"}),
        lambda: run_quality_gate(()),
        lambda: run_quality_gate((
            Check("duplicate", True, lambda: "ok"),
            Check("duplicate", False, lambda: "ok"),
        )),
        lambda: build_release_manifest(
            "python-course",
            version,
            "not-a-revision",
            files,
            built_at=timestamp,
        ),
        lambda: syntax_check({"notes.txt": "not Python"}),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError, QualityError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a deterministic quality gate and manifest verification demo."""
    run_self_checks()
    report = run_quality_gate(
        (
            Check("syntax", True, lambda: syntax_check({
                "src/course/core.py": "def ready() -> bool:\n    return True\n",
            })),
            Check("tests", True, lambda: "12 tests passed"),
            Check("coverage", False, lambda: "coverage report is advisory"),
        )
    )
    require_quality_gate(report)
    files = {
        "dist/python_course-2.4.0-py3-none-any.whl": b"deterministic wheel bytes",
        "dist/python_course-2.4.0.tar.gz": b"deterministic source bytes",
    }
    manifest = build_release_manifest(
        "python-course",
        parse_version("2.4.0"),
        "0123456789abcdef0123456789abcdef01234567",
        files,
        built_at=datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc),
    )
    print("Required checks passed:", report.passed)
    print("Advisory failures:", report.failed_advisory or "none")
    print("Artifact count:", len(json.loads(manifest)["artifacts"]))
    print("Release manifest digest:", _sha256(manifest.encode("utf-8")))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
