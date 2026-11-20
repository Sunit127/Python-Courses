"""Day 51: Disaster recovery, backups, restore testing, and RPO/RTO.

A backup is useful only when it can be restored within an agreed recovery
objective. This lesson models immutable full and incremental backups, verifies
their manifests, restores a selected point in time, and evaluates a recovery
drill. Everything runs in memory with the Python standard library: no files,
cloud accounts, databases, or networks are modified.

Learning goals
--------------
1. Distinguish a backup from replication and high availability.
2. Define recovery point objective (RPO) and recovery time objective (RTO).
3. Build verifiable full and incremental backup chains.
4. Restore into an isolated target and compare the result with expected data.
5. Turn restore-drill evidence into an explicit recovery readiness decision.

Teaching notes
--------------
- RPO is the maximum tolerable age of recovered data. If an incident occurs at
  12:00 and the newest usable backup represents 11:50, the observed data-loss
  window is ten minutes.
- RTO is the maximum tolerable time to restore service. Measure the entire
  recovery path: detection, access, restore, verification, and controlled
  cutover. A fast copy with no integrity check is not a successful recovery.
- Replication improves availability but can copy deletion or corruption to every
  replica. Backups should be isolated, access-controlled, immutable for a
  documented period, and protected by tested retention rules.
- A full backup contains a complete recovery point. An incremental backup stores
  changes and deletions since its parent. Every link is required for restore, so
  chain length and dependency health affect recovery time.
- Checksums detect accidental corruption; they do not prove who created an
  artifact. Production systems also need signed metadata, encryption, protected
  keys, authorization, audit logs, and geographic or account isolation.
- Never test a restore over the only production copy. Restore into an isolated
  target, verify schema and business invariants, scan for secrets in logs, then
  use a reviewed cutover procedure.
- A backup job saying "completed" is weak evidence. Schedule restore drills,
  record RPO/RTO, test lost credentials and missing dependencies, and assign
  owners to every failed objective.

Run this file with Python 3.10+. The self-checks cover complete and incremental
restores, deletions, corruption, broken chains, exact objective boundaries, and
invalid input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Iterable, Mapping


class RecoveryError(Exception):
    """Base class for expected backup and recovery failures."""


class ValidationError(RecoveryError):
    """Input cannot safely participate in backup or restore."""


class BackupNotFound(RecoveryError):
    """A requested backup does not exist in the catalog."""


class BrokenBackupChain(RecoveryError):
    """A required parent is missing or the chain contains a cycle."""


class IntegrityError(RecoveryError):
    """Backup metadata or content failed integrity verification."""


def require_aware(value: datetime, field_name: str) -> datetime:
    """Return value after rejecting non-datetime and naive timestamps."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field_name} must be timezone-aware")
    return value


def finite_non_negative(value: float, field_name: str) -> float:
    """Validate a finite, non-negative real number without accepting bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{field_name} must be finite and non-negative")
    return result


def clean_id(value: str, field_name: str) -> str:
    """Validate a short identifier used in immutable metadata."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValidationError(f"{field_name} must not be blank")
    if len(result) > 100:
        raise ValidationError(f"{field_name} is too long")
    return result


def normalize_path(value: str) -> str:
    """Return a safe, portable relative path for a backup entry."""
    path = clean_id(value, "path").replace("\\", "/")
    parts = path.split("/")
    if path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValidationError("path must be a normalized relative path")
    if ":" in parts[0]:
        raise ValidationError("path must not contain a drive or URI prefix")
    return "/".join(parts)


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class RecoveryObjective:
    """Maximum accepted data-loss window and recovery duration."""

    max_rpo: timedelta
    max_rto: timedelta
    max_chain_length: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.max_rpo, timedelta):
            raise TypeError("max_rpo must be a timedelta")
        if not isinstance(self.max_rto, timedelta):
            raise TypeError("max_rto must be a timedelta")
        if self.max_rpo < timedelta(0):
            raise ValidationError("max_rpo must not be negative")
        if self.max_rto <= timedelta(0):
            raise ValidationError("max_rto must be positive")
        if isinstance(self.max_chain_length, bool) or not isinstance(
            self.max_chain_length, int
        ):
            raise TypeError("max_chain_length must be an integer")
        if self.max_chain_length < 1:
            raise ValidationError("max_chain_length must be positive")


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One immutable file-like value in a recovery point."""

    path: str
    content: bytes
    checksum: str

    def __post_init__(self) -> None:
        normalized = normalize_path(self.path)
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")
        if not isinstance(self.checksum, str):
            raise TypeError("checksum must be a string")
        checksum = self.checksum.strip().lower()
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValidationError("checksum must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "path", normalized)
        object.__setattr__(self, "checksum", checksum)

    @classmethod
    def from_content(cls, path: str, content: bytes) -> FileEntry:
        """Build an entry whose checksum matches content."""
        return cls(path=path, content=content, checksum=sha256_bytes(content))

    def verify(self) -> None:
        """Raise when stored bytes do not match the recorded checksum."""
        if sha256_bytes(self.content) != self.checksum:
            raise IntegrityError(f"checksum mismatch for {self.path}")


class BackupKind(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"


def manifest_digest(
    backup_id: str,
    kind: BackupKind,
    created_at: datetime,
    parent_id: str | None,
    entries: Iterable[FileEntry],
    deleted_paths: Iterable[str],
) -> str:
    """Hash canonical metadata and content digests for one artifact."""
    document = {
        "backup_id": backup_id,
        "kind": kind.value,
        "created_at": created_at.astimezone(timezone.utc).isoformat(),
        "parent_id": parent_id,
        "entries": [
            {"path": entry.path, "checksum": entry.checksum}
            for entry in sorted(entries, key=lambda item: item.path)
        ],
        "deleted_paths": sorted(deleted_paths),
    }
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


@dataclass(frozen=True, slots=True)
class BackupArtifact:
    """An immutable full or incremental backup artifact."""

    backup_id: str
    kind: BackupKind
    created_at: datetime
    parent_id: str | None
    entries: tuple[FileEntry, ...]
    deleted_paths: tuple[str, ...]
    manifest_checksum: str

    def __post_init__(self) -> None:
        backup_id = clean_id(self.backup_id, "backup_id")
        created_at = require_aware(self.created_at, "created_at")
        if not isinstance(self.kind, BackupKind):
            raise TypeError("kind must be a BackupKind")
        parent_id = (
            None if self.parent_id is None else clean_id(self.parent_id, "parent_id")
        )
        if self.kind is BackupKind.FULL and parent_id is not None:
            raise ValidationError("a full backup cannot have a parent")
        if self.kind is BackupKind.INCREMENTAL and parent_id is None:
            raise ValidationError("an incremental backup requires a parent")
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple")
        if not isinstance(self.deleted_paths, tuple):
            raise TypeError("deleted_paths must be a tuple")

        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValidationError("entry paths must be unique")
        deleted = tuple(normalize_path(path) for path in self.deleted_paths)
        if len(deleted) != len(set(deleted)):
            raise ValidationError("deleted paths must be unique")
        if set(paths) & set(deleted):
            raise ValidationError("a path cannot be changed and deleted together")
        if self.kind is BackupKind.FULL and deleted:
            raise ValidationError("a full backup cannot contain deletions")

        object.__setattr__(self, "backup_id", backup_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda item: item.path)))
        object.__setattr__(self, "deleted_paths", tuple(sorted(deleted)))

    def verify(self) -> None:
        """Verify every entry and the canonical manifest."""
        for entry in self.entries:
            entry.verify()
        expected = manifest_digest(
            self.backup_id,
            self.kind,
            self.created_at,
            self.parent_id,
            self.entries,
            self.deleted_paths,
        )
        if expected != self.manifest_checksum:
            raise IntegrityError(f"manifest mismatch for backup {self.backup_id}")


def normalize_source(source: Mapping[str, bytes]) -> dict[str, bytes]:
    """Copy and validate a source snapshot."""
    if not isinstance(source, Mapping):
        raise TypeError("source must be a mapping")
    result: dict[str, bytes] = {}
    for raw_path, content in source.items():
        path = normalize_path(raw_path)
        if path in result:
            raise ValidationError(f"duplicate normalized path: {path}")
        if not isinstance(content, bytes):
            raise TypeError(f"content for {path} must be bytes")
        result[path] = content
    return result


class BackupCatalog:
    """Store immutable artifacts and restore complete recovery points."""

    def __init__(self) -> None:
        self._artifacts: dict[str, BackupArtifact] = {}

    def add(self, artifact: BackupArtifact) -> None:
        """Verify and add one artifact without replacing an existing ID."""
        if not isinstance(artifact, BackupArtifact):
            raise TypeError("artifact must be a BackupArtifact")
        artifact.verify()
        if artifact.backup_id in self._artifacts:
            raise ValidationError("backup_id already exists")
        if (
            artifact.parent_id is not None
            and artifact.parent_id not in self._artifacts
        ):
            raise BrokenBackupChain("incremental parent is not in the catalog")
        if artifact.parent_id is not None:
            parent = self._artifacts[artifact.parent_id]
            if artifact.created_at < parent.created_at:
                raise ValidationError("incremental backup predates its parent")
        self._artifacts[artifact.backup_id] = artifact

    def get(self, backup_id: str) -> BackupArtifact:
        """Return one artifact by ID."""
        key = clean_id(backup_id, "backup_id")
        try:
            return self._artifacts[key]
        except KeyError as error:
            raise BackupNotFound(key) from error

    def chain(self, backup_id: str) -> tuple[BackupArtifact, ...]:
        """Return a verified full-to-selected chain."""
        current = self.get(backup_id)
        reversed_chain: list[BackupArtifact] = []
        seen: set[str] = set()

        while True:
            if current.backup_id in seen:
                raise BrokenBackupChain("backup chain contains a cycle")
            seen.add(current.backup_id)
            current.verify()
            reversed_chain.append(current)
            if current.parent_id is None:
                break
            try:
                current = self._artifacts[current.parent_id]
            except KeyError as error:
                raise BrokenBackupChain(
                    f"missing parent {current.parent_id}"
                ) from error

        chain = tuple(reversed(reversed_chain))
        if chain[0].kind is not BackupKind.FULL:
            raise BrokenBackupChain("backup chain does not begin with a full backup")
        return chain

    def restore(self, backup_id: str) -> dict[str, bytes]:
        """Reconstruct and verify the selected recovery point."""
        state: dict[str, bytes] = {}
        for artifact in self.chain(backup_id):
            for path in artifact.deleted_paths:
                state.pop(path, None)
            for entry in artifact.entries:
                state[entry.path] = entry.content
        return state

    def newest_at_or_before(self, moment: datetime) -> BackupArtifact:
        """Select the newest usable artifact no later than moment."""
        moment = require_aware(moment, "moment")
        candidates = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.created_at <= moment
        ]
        if not candidates:
            raise BackupNotFound("no backup exists at or before the incident")
        candidates.sort(key=lambda item: (item.created_at, item.backup_id), reverse=True)
        for artifact in candidates:
            try:
                self.chain(artifact.backup_id)
            except (BrokenBackupChain, IntegrityError):
                continue
            return artifact
        raise BrokenBackupChain("no usable backup chain exists before the incident")


def build_full_backup(
    backup_id: str,
    created_at: datetime,
    source: Mapping[str, bytes],
) -> BackupArtifact:
    """Create a complete immutable recovery point."""
    backup_id = clean_id(backup_id, "backup_id")
    created_at = require_aware(created_at, "created_at")
    normalized = normalize_source(source)
    entries = tuple(
        FileEntry.from_content(path, content)
        for path, content in sorted(normalized.items())
    )
    checksum = manifest_digest(
        backup_id,
        BackupKind.FULL,
        created_at,
        None,
        entries,
        (),
    )
    return BackupArtifact(
        backup_id,
        BackupKind.FULL,
        created_at,
        None,
        entries,
        (),
        checksum,
    )


def build_incremental_backup(
    backup_id: str,
    created_at: datetime,
    parent_id: str,
    parent_state: Mapping[str, bytes],
    source: Mapping[str, bytes],
) -> BackupArtifact:
    """Create only changed values and deletions relative to parent_state."""
    backup_id = clean_id(backup_id, "backup_id")
    parent_id = clean_id(parent_id, "parent_id")
    created_at = require_aware(created_at, "created_at")
    previous = normalize_source(parent_state)
    current = normalize_source(source)

    changed = tuple(
        FileEntry.from_content(path, content)
        for path, content in sorted(current.items())
        if previous.get(path) != content
    )
    deleted = tuple(sorted(set(previous) - set(current)))
    checksum = manifest_digest(
        backup_id,
        BackupKind.INCREMENTAL,
        created_at,
        parent_id,
        changed,
        deleted,
    )
    return BackupArtifact(
        backup_id,
        BackupKind.INCREMENTAL,
        created_at,
        parent_id,
        changed,
        deleted,
        checksum,
    )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Comparison between a restored point and its expected state."""

    matching_paths: int
    missing_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_paths
            or self.unexpected_paths
            or self.changed_paths
        )


def verify_restore(
    restored: Mapping[str, bytes],
    expected: Mapping[str, bytes],
) -> VerificationReport:
    """Compare bytes and path membership without trusting counts alone."""
    actual = normalize_source(restored)
    wanted = normalize_source(expected)
    common = set(actual) & set(wanted)
    changed = tuple(
        sorted(path for path in common if actual[path] != wanted[path])
    )
    matching = sum(actual[path] == wanted[path] for path in common)
    return VerificationReport(
        matching_paths=matching,
        missing_paths=tuple(sorted(set(wanted) - set(actual))),
        unexpected_paths=tuple(sorted(set(actual) - set(wanted))),
        changed_paths=changed,
    )


@dataclass(frozen=True, slots=True)
class RecoveryDrillReport:
    """Evidence from selecting, restoring, and verifying one recovery point."""

    incident_at: datetime
    selected_backup_id: str
    backup_at: datetime
    chain_length: int
    restore_seconds: float
    verification: VerificationReport

    def __post_init__(self) -> None:
        incident_at = require_aware(self.incident_at, "incident_at")
        backup_at = require_aware(self.backup_at, "backup_at")
        if backup_at > incident_at:
            raise ValidationError("backup_at cannot be after incident_at")
        clean_id(self.selected_backup_id, "selected_backup_id")
        if isinstance(self.chain_length, bool) or not isinstance(
            self.chain_length, int
        ):
            raise TypeError("chain_length must be an integer")
        if self.chain_length < 1:
            raise ValidationError("chain_length must be positive")
        seconds = finite_non_negative(self.restore_seconds, "restore_seconds")
        object.__setattr__(self, "incident_at", incident_at)
        object.__setattr__(self, "backup_at", backup_at)
        object.__setattr__(self, "restore_seconds", seconds)

    @property
    def observed_rpo(self) -> timedelta:
        return self.incident_at - self.backup_at

    @property
    def observed_rto(self) -> timedelta:
        return timedelta(seconds=self.restore_seconds)


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    """An explainable go/no-go result for recovery readiness."""

    ready: bool
    reasons: tuple[str, ...]


def evaluate_readiness(
    report: RecoveryDrillReport,
    objective: RecoveryObjective,
) -> ReadinessDecision:
    """Evaluate RPO, RTO, chain length, and data verification."""
    if not isinstance(report, RecoveryDrillReport):
        raise TypeError("report must be a RecoveryDrillReport")
    if not isinstance(objective, RecoveryObjective):
        raise TypeError("objective must be a RecoveryObjective")

    failures: list[str] = []
    if report.observed_rpo > objective.max_rpo:
        failures.append("observed RPO exceeds the recovery objective")
    if report.observed_rto > objective.max_rto:
        failures.append("observed RTO exceeds the recovery objective")
    if report.chain_length > objective.max_chain_length:
        failures.append("backup chain is longer than the tested policy permits")
    if not report.verification.passed:
        failures.append("restored data does not match the expected recovery point")
    if failures:
        return ReadinessDecision(False, tuple(failures))
    return ReadinessDecision(
        True,
        ("restore drill meets RPO, RTO, chain, and integrity objectives",),
    )


def run_recovery_drill(
    catalog: BackupCatalog,
    incident_at: datetime,
    expected_state: Mapping[str, bytes],
    restore_seconds: float,
) -> RecoveryDrillReport:
    """Select the newest usable point, restore it, and record evidence."""
    if not isinstance(catalog, BackupCatalog):
        raise TypeError("catalog must be a BackupCatalog")
    incident_at = require_aware(incident_at, "incident_at")
    selected = catalog.newest_at_or_before(incident_at)
    chain = catalog.chain(selected.backup_id)
    restored = catalog.restore(selected.backup_id)
    verification = verify_restore(restored, expected_state)
    return RecoveryDrillReport(
        incident_at=incident_at,
        selected_backup_id=selected.backup_id,
        backup_at=selected.created_at,
        chain_length=len(chain),
        restore_seconds=restore_seconds,
        verification=verification,
    )


def utc(hour: int, minute: int = 0) -> datetime:
    """Build a deterministic timestamp for the examples."""
    return datetime(2026, 11, 20, hour, minute, tzinfo=timezone.utc)


def example_catalog() -> tuple[BackupCatalog, dict[str, bytes]]:
    """Build one full backup and two incremental recovery points."""
    catalog = BackupCatalog()
    state_1 = {
        "db/orders.json": b'[{"id": 1, "status": "new"}]',
        "config/schema.txt": b"version=3\n",
        "uploads/readme.txt": b"customer attachment\n",
    }
    full = build_full_backup("full-0800", utc(8), state_1)
    catalog.add(full)

    state_2 = {
        **state_1,
        "db/orders.json": b'[{"id": 1, "status": "paid"}]',
        "db/payments.json": b'[{"order_id": 1, "amount": 2500}]',
    }
    inc_1 = build_incremental_backup(
        "inc-0830", utc(8, 30), full.backup_id, catalog.restore(full.backup_id), state_2
    )
    catalog.add(inc_1)

    state_3 = {
        "db/orders.json": state_2["db/orders.json"],
        "db/payments.json": state_2["db/payments.json"],
        "config/schema.txt": b"version=4\n",
    }
    inc_2 = build_incremental_backup(
        "inc-0900", utc(9), inc_1.backup_id, catalog.restore(inc_1.backup_id), state_3
    )
    catalog.add(inc_2)
    return catalog, state_3


def run_self_checks() -> None:
    """Exercise success paths, boundaries, corruption, and invalid input."""
    catalog, expected = example_catalog()
    restored = catalog.restore("inc-0900")
    assert restored == expected
    assert "uploads/readme.txt" not in restored
    assert len(catalog.chain("inc-0900")) == 3

    verification = verify_restore(restored, expected)
    assert verification.passed
    assert verification.matching_paths == 3

    report = run_recovery_drill(
        catalog,
        incident_at=utc(9, 15),
        expected_state=expected,
        restore_seconds=300,
    )
    assert report.selected_backup_id == "inc-0900"
    assert report.observed_rpo == timedelta(minutes=15)
    assert report.observed_rto == timedelta(minutes=5)

    exact_objective = RecoveryObjective(
        max_rpo=timedelta(minutes=15),
        max_rto=timedelta(minutes=5),
        max_chain_length=3,
    )
    exact_decision = evaluate_readiness(report, exact_objective)
    assert exact_decision.ready

    strict_objective = RecoveryObjective(
        max_rpo=timedelta(minutes=10),
        max_rto=timedelta(minutes=4),
        max_chain_length=2,
    )
    strict_decision = evaluate_readiness(report, strict_objective)
    assert not strict_decision.ready
    assert len(strict_decision.reasons) == 3

    wrong_expected = {**expected, "db/orders.json": b"corrupted expectation"}
    mismatch = run_recovery_drill(
        catalog, utc(9, 15), wrong_expected, restore_seconds=60
    )
    assert not mismatch.verification.passed
    assert mismatch.verification.changed_paths == ("db/orders.json",)
    assert not evaluate_readiness(mismatch, exact_objective).ready

    good_entry = FileEntry.from_content("safe/data.txt", b"original")
    corrupt_entry = FileEntry(
        good_entry.path, b"tampered", good_entry.checksum
    )
    corrupt = BackupArtifact(
        backup_id="corrupt",
        kind=BackupKind.FULL,
        created_at=utc(7),
        parent_id=None,
        entries=(corrupt_entry,),
        deleted_paths=(),
        manifest_checksum=manifest_digest(
            "corrupt", BackupKind.FULL, utc(7), None, (corrupt_entry,), ()
        ),
    )
    try:
        catalog.add(corrupt)
    except IntegrityError:
        pass
    else:
        raise AssertionError("corrupt content must be rejected")

    try:
        catalog.get("missing")
    except BackupNotFound:
        pass
    else:
        raise AssertionError("an unknown backup must be rejected")

    invalid_calls = (
        lambda: normalize_path("../secret"),
        lambda: normalize_source({"data.txt": "not bytes"}),
        lambda: RecoveryObjective(timedelta(0), timedelta(0)),
        lambda: FileEntry.from_content("/absolute.txt", b"x"),
        lambda: run_recovery_drill(catalog, utc(9, 15), expected, float("nan")),
        lambda: build_incremental_backup(
            "bad", utc(8), "", {}, {}
        ),
        lambda: catalog.add(
            build_incremental_backup(
                "orphan", utc(10), "missing-parent", {}, {}
            )
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, RecoveryError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def demo() -> None:
    """Run a deterministic restore drill and print its decision."""
    catalog, expected = example_catalog()
    objective = RecoveryObjective(
        max_rpo=timedelta(minutes=30),
        max_rto=timedelta(minutes=10),
        max_chain_length=4,
    )
    report = run_recovery_drill(
        catalog,
        incident_at=utc(9, 20),
        expected_state=expected,
        restore_seconds=420,
    )
    decision = evaluate_readiness(report, objective)

    print(f"Selected backup: {report.selected_backup_id}")
    print(f"Backup chain length: {report.chain_length}")
    print(f"Observed RPO: {report.observed_rpo}")
    print(f"Observed RTO: {report.observed_rto}")
    print(f"Integrity verification passed: {report.verification.passed}")
    print(f"Recovery ready: {decision.ready}")
    print(f"Reason: {decision.reasons[0]}")


# Practice exercises
# ------------------
# 1. Add a retention policy that preserves a minimum number of recent full
#    backups while expiring old incremental chains as one unit.
# 2. Add an encryption_key_id to BackupArtifact metadata. Reject blank IDs and
#    write a report that lists artifacts using a key scheduled for retirement.
# 3. Add a database schema version to every recovery point. Reject a cutover when
#    the restored version is newer than the application can read.
# 4. Model detection, credential retrieval, restore, verification, and cutover as
#    separate drill phases. Report their individual and total durations.
# 5. Add a second storage location and evaluate whether every full backup has a
#    verified copy outside the primary failure domain.

# Solutions / clear guidance
# --------------------------
# 1. Group each full backup with all descendants until the next full backup.
#    Select whole groups for deletion; never leave incrementals whose parent was
#    removed. Test exact retention boundaries and an active legal hold.
# 2. Put only the non-secret key identifier in metadata. Encryption and key
#    retrieval belong behind separate interfaces. A rotation report should show
#    backup ID, key ID, age, and the last successful restore drill.
# 3. Treat schema compatibility as a cutover gate, not as a restore transport
#    error. Test equal, older-compatible, older-needing-migration, and future
#    versions without modifying the only restored copy.
# 4. Store immutable phase observations with monotonic durations. Sum every phase
#    for RTO and retain UTC timestamps for the audit timeline.
# 5. Label storage by failure domain, not merely by vendor name. Verify checksum,
#    credentials, and restore access independently; two buckets in one account
#    may share the same deletion or credential risk.

# Expert challenge / mini-project
# --------------------------------
# Build a local disaster-recovery drill runner around these primitives:
#
# * Load a reviewed recovery plan containing owners, dependencies, RPO/RTO,
#   backup locations, schema compatibility, and a controlled cutover checklist.
# * Discover backup metadata from fake adapters only, verify signatures and
#   checksums, select a point in time, and explain every rejected candidate.
# * Restore into a temporary isolated directory or disposable test database.
#   Prevent absolute paths, traversal, symlink escapes, and overwriting existing
#   targets. Bound file count, individual size, and total restored bytes.
# * Run technical checks plus business invariants: row counts, foreign keys,
#   balanced totals, required configuration, and a read-only application smoke
#   test. Do not expose restored personal data in logs.
# * Simulate a missing parent, corrupt artifact, unavailable key, revoked
#   credentials, region outage, slow restore, and failed verification. Every
#   failure must leave production untouched and produce an actionable report.
# * Persist an append-only drill record with plan hash, artifact IDs, operators,
#   phase durations, observed RPO/RTO, verification evidence, decision, owners,
#   and follow-up due dates.
#
# A strong solution keeps selection and policy pure, places storage, encryption,
# database, and notification operations behind narrow adapters, and defaults to
# offline simulation. Add unit tests, property tests for path normalization, and
# an end-to-end restore drill in CI using synthetic non-sensitive data. A green
# backup job never substitutes for a recent successful restore.


if __name__ == "__main__":
    run_self_checks()
    demo()
