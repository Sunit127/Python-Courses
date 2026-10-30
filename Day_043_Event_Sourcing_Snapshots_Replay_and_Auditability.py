"""Day 43: Event sourcing snapshots, replay, and auditability.

Learning goals
--------------
1. Represent an aggregate as an append-only stream of immutable domain events.
2. Rehydrate state deterministically by folding events in sequence order.
3. Use validated snapshots to bound replay time without replacing the event log.
4. Export a tamper-evident audit trail with a per-stream hash chain.
5. Design idempotent commands, optimistic concurrency, and recovery checks.

Teaching notes
--------------
- Event sourcing stores facts ("MoneyDeposited"), not a mutable current row.
  The current state is a projection produced by replay. It is a persistence
  strategy, not a requirement for every service.
- A command validates business rules against rehydrated state, then appends a
  batch with an expected stream version. A stale writer gets a conflict rather
  than silently losing an update.
- A snapshot is only a checkpoint. It records state after one exact sequence
  and the hash of that event. On load, validate the anchor and replay the
  suffix; if the snapshot is invalid, safely replay from the beginning.
- Replay must be deterministic: no clocks, randomness, network calls, or
  mutable globals in the fold. Store nondeterministic facts in event data.
- Hash chains detect accidental edits and reorderings. They do not replace
  encryption, access control, signatures, retention policy, or redaction.
- Production events usually carry schema version, actor, correlation id, and
  causation id. Keep secrets and unnecessary personal data out of the log.

Run with Python 3.10+.

Practice exercises
------------------
1. Implement count_event_kinds(events) to return kind counts and reject
   malformed values.
2. Implement replay_after_snapshot(snapshot, events) to validate the anchor
   and return the event suffix.
3. Implement audit_digest(events) using the final hash, or "" for no events.

Solutions appear in the executable code below.

Expert challenge: production event-sourced service
---------------------------------------------------
Build a versioned order aggregate with snapshots and two read projections.
Persist command deduplication keys transactionally; add schema upcasters,
encrypted sensitive fields, snapshot compaction, replay checkpoints, a
dead-letter path for poison events, and metrics for replay duration, snapshot
hit rate, and hash failures. Document retention, GDPR redaction boundaries,
recovery objectives, and how an auditor verifies a released report.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json


class EventSourcingError(RuntimeError):
    """Base class for event-log and replay failures."""


class ValidationError(EventSourcingError, ValueError):
    """Raised for invalid public input."""


class ConcurrencyConflict(EventSourcingError):
    """Raised when an append uses a stale stream version."""


class DuplicateCommand(EventSourcingError):
    """Raised when a command id is reused for different work."""


class AuditFailure(EventSourcingError):
    """Raised when an event hash chain or snapshot anchor is invalid."""


class AccountNotFound(EventSourcingError):
    """Raised when a command targets an unknown account."""


class AccountStateError(EventSourcingError):
    """Raised when an account invariant would be violated."""


def _text(value: object, *, field: str, limit: int = 120) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValidationError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise ValidationError(f"{field} must be at most {limit} characters")
    return cleaned


def _positive_cents(value: object, *, field: str = "amount_cents") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _non_negative(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field} must be a non-negative integer")
    return value


def _canonical(data: Mapping[str, object]) -> str:
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise ValidationError("event data must be JSON-safe") from error


def _hash_event(previous: str, aggregate_id: str, sequence: int, kind: str,
                data: Mapping[str, object]) -> str:
    material = "|".join((previous, aggregate_id, str(sequence), kind, _canonical(data)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    aggregate_id: str
    sequence: int
    kind: str
    data: Mapping[str, object]
    previous_hash: str
    event_hash: str


@dataclass(frozen=True, slots=True)
class AccountState:
    account_id: str
    owner: str
    balance_cents: int
    status: str
    version: int
    last_hash: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    account_id: str
    version: int
    state: AccountState
    event_hash: str


def _normalize_changes(
    changes: Iterable[tuple[str, Mapping[str, object]]],
) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
    try:
        raw = tuple(changes)
    except TypeError as error:
        raise ValidationError("changes must be iterable") from error
    if not raw:
        raise ValidationError("changes must not be empty")
    normalized: list[tuple[str, tuple[tuple[str, object], ...]]] = []
    for kind, data in raw:
        clean_kind = _text(kind, field="event kind", limit=80)
        if not isinstance(data, Mapping):
            raise ValidationError("event data must be a mapping")
        copied = dict(data)
        _canonical(copied)
        normalized.append((clean_kind, tuple(sorted(copied.items()))))
    return tuple(normalized)


class EventLog:
    """In-memory append-only streams with command idempotency."""

    def __init__(self) -> None:
        self._streams: dict[str, list[Event]] = {}
        self._commands: dict[
            str, tuple[str, tuple[tuple[str, tuple[tuple[str, object], ...]], ...]]
        ] = {}

    def version(self, aggregate_id: str) -> int:
        clean_id = _text(aggregate_id, field="aggregate_id")
        return len(self._streams.get(clean_id, ()))

    def stream(self, aggregate_id: str) -> tuple[Event, ...]:
        clean_id = _text(aggregate_id, field="aggregate_id")
        return tuple(self._streams.get(clean_id, ()))

    def append(
        self,
        aggregate_id: str,
        *,
        expected_version: int,
        command_id: str,
        changes: Iterable[tuple[str, Mapping[str, object]]],
    ) -> tuple[Event, ...]:
        clean_id = _text(aggregate_id, field="aggregate_id")
        clean_command = _text(command_id, field="command_id")
        expected = _non_negative(expected_version, field="expected_version")
        fingerprint = _normalize_changes(changes)
        previous = self._commands.get(clean_command)
        if previous is not None:
            old_id, old_fingerprint = previous
            if old_id != clean_id or old_fingerprint != fingerprint:
                raise DuplicateCommand("command id was reused for different work")
            return tuple(
                event for event in self._streams[clean_id]
                if event.event_id.startswith(f"{clean_command}:")
            )
        current = self.version(clean_id)
        if current != expected:
            raise ConcurrencyConflict(f"{clean_id!r} is at version {current}, expected {expected}")

        stream = self._streams.setdefault(clean_id, [])
        previous_hash = "" if not stream else stream[-1].event_hash
        appended: list[Event] = []
        for offset, (kind, pairs) in enumerate(fingerprint, start=1):
            sequence = current + offset
            data = dict(pairs)
            digest = _hash_event(previous_hash, clean_id, sequence, kind, data)
            event = Event(
                f"{clean_command}:{offset}", clean_id, sequence, kind, data,
                previous_hash, digest
            )
            stream.append(event)
            appended.append(event)
            previous_hash = digest
        self._commands[clean_command] = (clean_id, fingerprint)
        return tuple(appended)

    def verify_chain(self, aggregate_id: str) -> None:
        """Raise AuditFailure if a stream was edited or reordered."""
        previous_hash = ""
        for expected_sequence, event in enumerate(self.stream(aggregate_id), start=1):
            if event.sequence != expected_sequence:
                raise AuditFailure("event sequence is not contiguous")
            if event.previous_hash != previous_hash:
                raise AuditFailure("event predecessor hash does not match")
            expected_hash = _hash_event(
                event.previous_hash, event.aggregate_id, event.sequence,
                event.kind, event.data
            )
            if event.event_hash != expected_hash:
                raise AuditFailure(f"event {event.event_id!r} failed hash verification")
            previous_hash = event.event_hash

    def audit_lines(self, aggregate_id: str) -> tuple[str, ...]:
        """Return stable JSON lines suitable for an audit export."""
        self.verify_chain(aggregate_id)
        return tuple(
            _canonical({
                "event_id": event.event_id,
                "aggregate_id": event.aggregate_id,
                "sequence": event.sequence,
                "kind": event.kind,
                "data": dict(event.data),
                "previous_hash": event.previous_hash,
                "event_hash": event.event_hash,
            })
            for event in self.stream(aggregate_id)
        )


class SnapshotStore:
    """A replaceable store for validated snapshots."""

    def __init__(self) -> None:
        self._snapshots: dict[str, Snapshot] = {}

    def save(self, snapshot: Snapshot) -> None:
        if not isinstance(snapshot, Snapshot):
            raise ValidationError("snapshot must be Snapshot")
        self._snapshots[snapshot.account_id] = snapshot

    def get(self, account_id: str) -> Snapshot | None:
        clean_id = _text(account_id, field="account_id")
        return self._snapshots.get(clean_id)


def _initial(account_id: str) -> AccountState:
    return AccountState(account_id, "", 0, "missing", 0, "")


def apply_event(state: AccountState, event: Event) -> AccountState:
    """Pure fold step: equal inputs always produce equal outputs."""
    if not isinstance(state, AccountState) or not isinstance(event, Event):
        raise ValidationError("state and event types are invalid")
    if event.aggregate_id != state.account_id:
        raise ValidationError("event belongs to another aggregate")
    if event.sequence != state.version + 1:
        raise AuditFailure("event sequence does not follow state version")

    if event.kind == "AccountOpened":
        if state.status != "missing":
            raise AccountStateError("account is already open or closed")
        owner = _text(event.data.get("owner"), field="owner")
        return replace(state, owner=owner, status="open",
                       version=event.sequence, last_hash=event.event_hash)

    if event.kind in {"MoneyDeposited", "MoneyWithdrawn"}:
        if state.status != "open":
            raise AccountStateError("account is not open")
        amount = _positive_cents(event.data.get("amount_cents"))
        delta = amount if event.kind == "MoneyDeposited" else -amount
        balance = state.balance_cents + delta
        if balance < 0:
            raise AccountStateError("balance cannot become negative")
        return replace(state, balance_cents=balance,
                       version=event.sequence, last_hash=event.event_hash)

    if event.kind == "AccountClosed":
        if state.status != "open" or state.balance_cents:
            raise AccountStateError("account must be open with zero balance")
        return replace(state, status="closed",
                       version=event.sequence, last_hash=event.event_hash)

    raise ValidationError(f"unsupported event kind: {event.kind}")


def replay(events: Iterable[Event], *, initial: AccountState) -> AccountState:
    """Fold a stream, starting after an optional snapshot."""
    if not isinstance(initial, AccountState):
        raise ValidationError("initial must be AccountState")
    try:
        items = tuple(events)
    except TypeError as error:
        raise ValidationError("events must be iterable") from error
    state = initial
    for event in items:
        state = apply_event(state, event)
    return state


class AccountRepository:
    """Load state from a snapshot plus the remaining event suffix."""

    def __init__(self, log: EventLog, snapshots: SnapshotStore) -> None:
        if not isinstance(log, EventLog) or not isinstance(snapshots, SnapshotStore):
            raise ValidationError("log and snapshots have invalid types")
        self._log, self._snapshots = log, snapshots

    def load(self, account_id: str) -> AccountState:
        clean_id = _text(account_id, field="account_id")
        self._log.verify_chain(clean_id)
        events = self._log.stream(clean_id)
        snapshot = self._snapshots.get(clean_id)
        initial, start = _initial(clean_id), 0

        if snapshot is not None:
            valid = (
                snapshot.account_id == clean_id
                and snapshot.version == snapshot.state.version
                and snapshot.version <= len(events)
                and snapshot.event_hash == snapshot.state.last_hash
            )
            if valid and snapshot.version:
                valid = events[snapshot.version - 1].event_hash == snapshot.event_hash
            if valid:
                initial, start = snapshot.state, snapshot.version

        if not events and initial.status == "missing":
            raise AccountNotFound(f"account {clean_id!r} does not exist")
        state = replay(events[start:], initial=initial)
        if state.status == "missing":
            raise AccountNotFound(f"account {clean_id!r} does not exist")
        return state

    def save_snapshot(self, account_id: str) -> Snapshot:
        state = self.load(account_id)
        snapshot = Snapshot(state.account_id, state.version, state, state.last_hash)
        self._snapshots.save(snapshot)
        return snapshot


class AccountService:
    """Command side: validate invariants, then append facts."""

    def __init__(self, log: EventLog, repository: AccountRepository) -> None:
        if not isinstance(log, EventLog) or not isinstance(repository, AccountRepository):
            raise ValidationError("log and repository have invalid types")
        self._log, self._repository = log, repository

    def open_account(self, account_id: str, owner: str, *, command_id: str) -> tuple[Event, ...]:
        clean_id, clean_owner = _text(account_id, field="account_id"), _text(owner, field="owner")
        if self._log.version(clean_id):
            raise AccountStateError("account already exists")
        return self._log.append(
            clean_id, expected_version=0, command_id=command_id,
            changes=(("AccountOpened", {"owner": clean_owner}),)
        )

    def deposit(self, account_id: str, amount_cents: int, *, command_id: str) -> tuple[Event, ...]:
        clean_id, amount = _text(account_id, field="account_id"), _positive_cents(amount_cents)
        state = self._repository.load(clean_id)
        if state.status != "open":
            raise AccountStateError("account is not open")
        return self._log.append(
            clean_id, expected_version=state.version, command_id=command_id,
            changes=(("MoneyDeposited", {"amount_cents": amount}),)
        )

    def withdraw(self, account_id: str, amount_cents: int, *, command_id: str) -> tuple[Event, ...]:
        clean_id, amount = _text(account_id, field="account_id"), _positive_cents(amount_cents)
        state = self._repository.load(clean_id)
        if state.status != "open":
            raise AccountStateError("account is not open")
        if amount > state.balance_cents:
            raise AccountStateError("insufficient funds")
        return self._log.append(
            clean_id, expected_version=state.version, command_id=command_id,
            changes=(("MoneyWithdrawn", {"amount_cents": amount}),)
        )

    def close(self, account_id: str, *, command_id: str) -> tuple[Event, ...]:
        clean_id = _text(account_id, field="account_id")
        state = self._repository.load(clean_id)
        if state.status != "open" or state.balance_cents:
            raise AccountStateError("account must be open with zero balance")
        return self._log.append(
            clean_id, expected_version=state.version, command_id=command_id,
            changes=(("AccountClosed", {}),)
        )


def count_event_kinds(events: Iterable[Event]) -> dict[str, int]:
    """Solution 1."""
    try:
        items = tuple(events)
    except TypeError as error:
        raise ValidationError("events must be iterable") from error
    if any(not isinstance(event, Event) for event in items):
        raise ValidationError("events must contain Event values")
    return dict(Counter(event.kind for event in items))


def replay_after_snapshot(snapshot: Snapshot, events: Iterable[Event]) -> tuple[Event, ...]:
    """Solution 2."""
    if not isinstance(snapshot, Snapshot):
        raise ValidationError("snapshot must be Snapshot")
    try:
        items = tuple(events)
    except TypeError as error:
        raise ValidationError("events must be iterable") from error
    if any(not isinstance(event, Event) for event in items):
        raise ValidationError("events must contain Event values")
    if snapshot.version > len(items) or snapshot.version != snapshot.state.version:
        raise AuditFailure("snapshot version is invalid")
    if snapshot.version and items[snapshot.version - 1].event_hash != snapshot.event_hash:
        raise AuditFailure("snapshot anchor does not match the stream")
    return tuple(event for event in items if event.sequence > snapshot.version)


def audit_digest(events: Iterable[Event]) -> str:
    """Solution 3."""
    try:
        items = tuple(events)
    except TypeError as error:
        raise ValidationError("events must be iterable") from error
    if any(not isinstance(event, Event) for event in items):
        raise ValidationError("events must contain Event values")
    return "" if not items else items[-1].event_hash


def _expect_error(callback: object, expected: type[BaseException]) -> None:
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        callback()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def run_self_checks() -> None:
    """Check replay, snapshots, idempotency, concurrency, and tamper detection."""
    log, snapshots = EventLog(), SnapshotStore()
    repository = AccountRepository(log, snapshots)
    service = AccountService(log, repository)

    opened = service.open_account("acct-43", " Ada Lovelace ", command_id="open-43")
    assert repository.load("acct-43").balance_cents == 0
    service.deposit("acct-43", 1_000, command_id="deposit-43")
    service.deposit("acct-43", 500, command_id="deposit-44")
    checkpoint = repository.save_snapshot("acct-43")
    assert checkpoint.version == 3 and checkpoint.state.balance_cents == 1_500
    assert replay_after_snapshot(checkpoint, log.stream("acct-43")) == ()
    assert count_event_kinds(log.stream("acct-43")) == {
        "AccountOpened": 1, "MoneyDeposited": 2
    }
    assert audit_digest(log.stream("acct-43")) == log.stream("acct-43")[-1].event_hash
    assert len(log.audit_lines("acct-43")) == 3

    assert service.deposit("acct-43", 500, command_id="deposit-44") == log.stream("acct-43")[2:3]
    _expect_error(
        lambda: log.append(
            "acct-43", expected_version=2, command_id="stale",
            changes=(("MoneyDeposited", {"amount_cents": 1}),)
        ),
        ConcurrencyConflict,
    )
    _expect_error(
        lambda: service.deposit("acct-43", 1, command_id="deposit-44"),
        DuplicateCommand,
    )
    _expect_error(
        lambda: service.withdraw("acct-43", 2_000, command_id="too-much"),
        AccountStateError,
    )

    service.withdraw("acct-43", 1_500, command_id="withdraw-43")
    service.close("acct-43", command_id="close-43")
    assert repository.load("acct-43").status == "closed"

    snapshots.save(replace(checkpoint, event_hash="tampered"))
    assert repository.load("acct-43").status == "closed"

    log._streams["acct-43"][0] = replace(log.stream("acct-43")[0], data={"owner": "Mallory"})
    _expect_error(lambda: log.verify_chain("acct-43"), AuditFailure)


def main() -> None:
    """Demonstrate snapshot reuse and stable audit export."""
    run_self_checks()
    log, snapshots = EventLog(), SnapshotStore()
    repository = AccountRepository(log, snapshots)
    service = AccountService(log, repository)
    service.open_account("demo-43", "Grace Hopper", command_id="demo-open")
    service.deposit("demo-43", 4_200, command_id="demo-deposit")
    checkpoint = repository.save_snapshot("demo-43")
    service.withdraw("demo-43", 700, command_id="demo-withdraw")
    print("Snapshot version:", checkpoint.version)
    print("Replayed state:", repository.load("demo-43"))
    print("Audit digest:", audit_digest(log.stream("demo-43"))[:16] + "...")
    print("Audit lines:", len(log.audit_lines("demo-43")))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
