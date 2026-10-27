"""Day 42: CQRS, materialized views, and eventual-consistency projections.

Learning goals
--------------
1. Separate command (write) models from query (read) models when their
   responsibilities and scaling needs differ.
2. Append immutable domain events with optimistic version checks.
3. Build a materialized view that can lag, catch up, ignore duplicate delivery,
   and rebuild by replaying the event stream.
4. Make consistency guarantees explicit instead of hiding stale-read behavior.
5. Choose idempotency keys, projection checkpoints, and recovery policies for
   production event-driven systems.

Teaching notes
--------------
- CQRS means Command Query Responsibility Segregation, not "use two databases"
  by default. A command model protects invariants; a query model is shaped for
  fast, safe reads. Start with one store if that is enough.
- An event is a durable fact ("MoneyDeposited"), not an instruction ("deposit
  now"). Events are append-only. A stream version lets two writers detect a
  lost update with an optimistic-concurrency check.
- A projection is a derived read model. It may be milliseconds behind the
  command store, so APIs should label reads as eventually consistent or offer a
  read-your-writes option. Never let a stale projection enforce write rules.
- At-least-once delivery means a projection can see the same event twice.
  Applying an event must therefore be idempotent (track event ids or use a
  unique constraint). A checkpoint makes restart work bounded and observable.
- Rebuilds are a normal operation: deploy a new projection version, replay the
  stream, compare it with the old view, and switch reads only after validation.
  In production, snapshots can shorten replay while preserving the event log.
- This file uses only the standard library and an in-memory store. It does not
  pretend to provide durability, transactions across services, or a message
  broker. Those are the expert challenge.

Run this file with Python 3.10+.

Practice exercises
------------------
1. Implement fold_balance(events) so it accepts only valid money events,
   rejects a negative balance, and returns cents.
2. Implement unseen_event_ids(events, applied_ids) to return new event ids
   in delivery order without duplicates.
3. Implement projection_lag(produced_version, applied_version) with
   validation for non-negative integers.

Solutions appear in the executable code below.

Expert challenge: production CQRS read model
---------------------------------------------
Build a versioned order service with an append-only event store and two
projections: a customer order summary and a revenue report. Persist command
deduplication keys and per-stream versions transactionally. Deliver events
through an at-least-once queue with retries and a dead-letter stream. Add
projection checkpoints, poison-event quarantine, schema upcasters, replay
metrics, lag-based alerts, and a blue/green projection rebuild. Document which
endpoints promise read-your-writes, which tolerate staleness, and how an
operator repairs a projection after a crash.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace


class CQRSFault(RuntimeError):
    """Base class for command, event-store, and projection failures."""


class ValidationError(CQRSFault, ValueError):
    """Raised when a public API receives invalid data."""


class ConcurrencyConflict(CQRSFault):
    """Raised when a writer uses a stale stream version."""


class DuplicateCommand(CQRSFault):
    """Raised when an idempotency key is reused with different work."""


class AccountNotFound(CQRSFault):
    """Raised when a command targets an unknown account."""


class AccountStateError(CQRSFault):
    """Raised when an operation violates an account invariant."""


class ProjectionGap(CQRSFault):
    """Raised when a projection receives an event out of sequence."""


def _text(value: object, *, field: str, limit: int = 100) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValidationError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise ValidationError(f"{field} must be at most {limit} characters")
    return cleaned


def _positive_cents(value: object, *, field: str = "amount_cents") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field} must be positive")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value < 0:
        raise ValidationError(f"{field} must not be negative")
    return value


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable fact in one aggregate stream."""

    event_id: str
    stream_id: str
    sequence: int
    kind: str
    data: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AccountView:
    """The query model returned to readers."""

    account_id: str
    owner: str
    balance_cents: int
    status: str
    version: int


@dataclass(frozen=True, slots=True)
class _AccountState:
    owner: str
    balance_cents: int
    status: str


class EventStore:
    """Small append-only store; a database would make append atomic."""

    def __init__(self) -> None:
        self._streams: dict[str, list[Event]] = {}
        self._commands: dict[str, tuple[str, tuple[tuple[str, tuple[tuple[str, object], ...]], ...]]] = {}
        self._global_events: list[Event] = []

    def version(self, stream_id: str) -> int:
        clean_id = _text(stream_id, field="stream_id")
        return len(self._streams.get(clean_id, ()))

    def stream(self, stream_id: str) -> tuple[Event, ...]:
        clean_id = _text(stream_id, field="stream_id")
        return tuple(self._streams.get(clean_id, ()))

    def all_events(self) -> tuple[Event, ...]:
        return tuple(self._global_events)

    def append(
        self,
        stream_id: str,
        *,
        expected_version: int,
        command_id: str,
        changes: Iterable[tuple[str, Mapping[str, object]]],
    ) -> tuple[Event, ...]:
        """Append facts once, using a stream-version compare-and-set."""
        clean_stream = _text(stream_id, field="stream_id")
        clean_command = _text(command_id, field="command_id")
        expected = _non_negative_int(expected_version, field="expected_version")
        normalized: list[tuple[str, tuple[tuple[str, object], ...]]] = []
        try:
            raw_changes = tuple(changes)
        except TypeError as error:
            raise ValidationError("changes must be iterable") from error
        if not raw_changes:
            raise ValidationError("changes must not be empty")
        for kind, data in raw_changes:
            clean_kind = _text(kind, field="event kind", limit=80)
            if not isinstance(data, Mapping):
                raise ValidationError("event data must be a mapping")
            copied = tuple(sorted((str(key), value) for key, value in data.items()))
            if any(not key for key, _ in copied):
                raise ValidationError("event data keys must not be empty")
            normalized.append((clean_kind, copied))
        fingerprint = tuple(normalized)

        previous = self._commands.get(clean_command)
        if previous is not None:
            previous_stream, previous_fingerprint = previous
            if previous_stream != clean_stream or previous_fingerprint != fingerprint:
                raise DuplicateCommand("command id was reused for different work")
            return tuple(
                event for event in self._streams[clean_stream]
                if event.event_id.startswith(f"{clean_command}:")
            )

        current = self.version(clean_stream)
        if current != expected:
            raise ConcurrencyConflict(
                f"{clean_stream!r} is at version {current}, expected {expected}"
            )

        stream = self._streams.setdefault(clean_stream, [])
        appended: list[Event] = []
        for offset, (kind, pairs) in enumerate(normalized, start=1):
            sequence = current + offset
            event = Event(
                event_id=f"{clean_command}:{offset}",
                stream_id=clean_stream,
                sequence=sequence,
                kind=kind,
                data=dict(pairs),
            )
            stream.append(event)
            self._global_events.append(event)
            appended.append(event)
        self._commands[clean_command] = (clean_stream, fingerprint)
        return tuple(appended)


class AccountCommandService:
    """Command side: enforce invariants from the event stream, not the read view."""

    def __init__(self, store: EventStore) -> None:
        if not isinstance(store, EventStore):
            raise ValidationError("store must be EventStore")
        self._store = store

    def _state(self, account_id: str) -> _AccountState:
        events = self._store.stream(account_id)
        if not events:
            raise AccountNotFound(f"account {account_id!r} does not exist")
        owner = ""
        balance = 0
        status = ""
        for event in events:
            if event.kind == "AccountOpened":
                owner = _text(event.data.get("owner"), field="owner")
                status = "open"
            elif event.kind == "MoneyDeposited":
                balance += _positive_cents(event.data.get("amount_cents"))
            elif event.kind == "MoneyWithdrawn":
                amount = _positive_cents(event.data.get("amount_cents"))
                balance -= amount
            elif event.kind == "AccountClosed":
                status = "closed"
        return _AccountState(owner, balance, status)

    def open_account(self, account_id: str, owner: str, *, command_id: str) -> tuple[Event, ...]:
        clean_id = _text(account_id, field="account_id")
        clean_owner = _text(owner, field="owner")
        if self._store.version(clean_id):
            raise AccountStateError("account already exists")
        return self._store.append(
            clean_id,
            expected_version=0,
            command_id=command_id,
            changes=(("AccountOpened", {"owner": clean_owner}),),
        )

    def deposit(self, account_id: str, amount_cents: int, *, command_id: str) -> tuple[Event, ...]:
        clean_id = _text(account_id, field="account_id")
        amount = _positive_cents(amount_cents)
        state = self._state(clean_id)
        if state.status != "open":
            raise AccountStateError("account is not open")
        return self._store.append(
            clean_id,
            expected_version=self._store.version(clean_id),
            command_id=command_id,
            changes=(("MoneyDeposited", {"amount_cents": amount}),),
        )

    def withdraw(self, account_id: str, amount_cents: int, *, command_id: str) -> tuple[Event, ...]:
        clean_id = _text(account_id, field="account_id")
        amount = _positive_cents(amount_cents)
        state = self._state(clean_id)
        if state.status != "open":
            raise AccountStateError("account is not open")
        if amount > state.balance_cents:
            raise AccountStateError("insufficient funds")
        return self._store.append(
            clean_id,
            expected_version=self._store.version(clean_id),
            command_id=command_id,
            changes=(("MoneyWithdrawn", {"amount_cents": amount}),),
        )

    def close(self, account_id: str, *, command_id: str) -> tuple[Event, ...]:
        clean_id = _text(account_id, field="account_id")
        state = self._state(clean_id)
        if state.status != "open":
            raise AccountStateError("account is not open")
        if state.balance_cents:
            raise AccountStateError("account must have a zero balance")
        return self._store.append(
            clean_id,
            expected_version=self._store.version(clean_id),
            command_id=command_id,
            changes=(("AccountClosed", {}),),
        )


class AccountProjection:
    """Query side: a replayable, eventually consistent materialized view."""

    def __init__(self) -> None:
        self._views: dict[str, AccountView] = {}
        self._seen_ids: set[str] = set()
        self._checkpoint = 0

    @property
    def checkpoint(self) -> int:
        """Number of globally ordered events successfully applied."""
        return self._checkpoint

    def get(self, account_id: str) -> AccountView | None:
        clean_id = _text(account_id, field="account_id")
        return self._views.get(clean_id)

    def apply(self, event: Event) -> None:
        """Apply one event; duplicate delivery is harmless."""
        if not isinstance(event, Event):
            raise ValidationError("event must be Event")
        if event.event_id in self._seen_ids:
            return
        current = self._views.get(event.stream_id)
        current_version = 0 if current is None else current.version
        if event.sequence != current_version + 1:
            raise ProjectionGap(
                f"stream {event.stream_id!r} expected {current_version + 1}, "
                f"received {event.sequence}"
            )

        if event.kind == "AccountOpened":
            if current is not None:
                raise AccountStateError("account is already projected")
            owner = _text(event.data.get("owner"), field="owner")
            view = AccountView(event.stream_id, owner, 0, "open", event.sequence)
        elif event.kind in {"MoneyDeposited", "MoneyWithdrawn"}:
            if current is None:
                raise ProjectionGap("money event arrived before AccountOpened")
            amount = _positive_cents(event.data.get("amount_cents"))
            delta = amount if event.kind == "MoneyDeposited" else -amount
            balance = current.balance_cents + delta
            if balance < 0:
                raise AccountStateError("projection would create a negative balance")
            view = replace(current, balance_cents=balance, version=event.sequence)
        elif event.kind == "AccountClosed":
            if current is None or current.status != "open" or current.balance_cents:
                raise AccountStateError("invalid close event for projection")
            view = replace(current, status="closed", version=event.sequence)
        else:
            raise ValidationError(f"unsupported event kind: {event.kind}")
        self._views[event.stream_id] = view
        self._seen_ids.add(event.event_id)
        self._checkpoint += 1

    def catch_up(self, store: EventStore, *, limit: int | None = None) -> int:
        """Apply a bounded batch and return how many events were consumed."""
        if not isinstance(store, EventStore):
            raise ValidationError("store must be EventStore")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValidationError("limit must be a positive integer")
        pending = store.all_events()[self._checkpoint :]
        if limit is not None:
            pending = pending[:limit]
        for event in pending:
            self.apply(event)
        return len(pending)

    def rebuild(self, store: EventStore) -> None:
        """Reset state and replay every event from the append-only log."""
        if not isinstance(store, EventStore):
            raise ValidationError("store must be EventStore")
        self._views.clear()
        self._seen_ids.clear()
        self._checkpoint = 0
        self.catch_up(store)


def fold_balance(events: Iterable[Event]) -> int:
    """Solution 1: fold only money events and reject invalid balances."""
    balance = 0
    try:
        items = tuple(events)
    except TypeError as error:
        raise ValidationError("events must be iterable") from error
    for event in items:
        if not isinstance(event, Event):
            raise ValidationError("events must contain Event values")
        if event.kind not in {"MoneyDeposited", "MoneyWithdrawn"}:
            raise ValidationError("fold_balance accepts only money events")
        amount = _positive_cents(event.data.get("amount_cents"))
        balance += amount if event.kind == "MoneyDeposited" else -amount
        if balance < 0:
            raise AccountStateError("balance cannot be negative")
    return balance


def unseen_event_ids(events: Iterable[Event], applied_ids: Iterable[str]) -> tuple[str, ...]:
    """Solution 2: return new ids in order while removing duplicates."""
    try:
        items = tuple(events)
        already = set(applied_ids)
    except (TypeError, ValueError) as error:
        raise ValidationError("events and applied_ids must be iterable") from error
    if any(not isinstance(event, Event) for event in items):
        raise ValidationError("events must contain Event values")
    if any(not isinstance(event_id, str) or not event_id for event_id in already):
        raise ValidationError("applied_ids must contain non-empty text")
    result: list[str] = []
    seen = set(already)
    for event in items:
        if event.event_id not in seen:
            result.append(event.event_id)
            seen.add(event.event_id)
    return tuple(result)


def projection_lag(produced_version: int, applied_version: int) -> int:
    """Solution 3: quantify staleness without allowing negative lag."""
    produced = _non_negative_int(produced_version, field="produced_version")
    applied = _non_negative_int(applied_version, field="applied_version")
    if applied > produced:
        raise ValidationError("applied_version cannot exceed produced_version")
    return produced - applied


def _expect_error(callback: object, expected: type[BaseException]) -> None:
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        callback()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def run_self_checks() -> None:
    """Verify command invariants, lag, idempotency, replay, and recovery."""
    store = EventStore()
    commands = AccountCommandService(store)
    projection = AccountProjection()

    opened = commands.open_account("acct-42", " Ada Lovelace ", command_id="open-42")
    assert opened[0].sequence == 1
    assert projection.get("acct-42") is None
    assert projection.catch_up(store) == 1
    assert projection.get("acct-42") == AccountView("acct-42", "Ada Lovelace", 0, "open", 1)

    deposited = commands.deposit("acct-42", 1_250, command_id="deposit-42")
    assert projection.get("acct-42").balance_cents == 0
    assert projection.catch_up(store, limit=1) == 1
    assert projection.get("acct-42").balance_cents == 1_250
    assert fold_balance(store.stream("acct-42")[1:]) == 1_250

    duplicate = commands.deposit("acct-42", 1_250, command_id="deposit-42")
    assert duplicate == deposited
    _expect_error(
        lambda: store.append(
            "acct-42",
            expected_version=store.version("acct-42") - 1,
            command_id="stale-command",
            changes=(("MoneyDeposited", {"amount_cents": 1}),),
        ),
        ConcurrencyConflict,
    )
    _expect_error(
        lambda: commands.withdraw("acct-42", 2_000, command_id="too-much"),
        AccountStateError,
    )

    commands.withdraw("acct-42", 1_250, command_id="withdraw-42")
    assert projection.catch_up(store) == 1
    assert projection.get("acct-42").balance_cents == 0
    closed = commands.close("acct-42", command_id="close-42")
    assert projection.get("acct-42").status == "open"
    projection.catch_up(store)
    assert projection.get("acct-42").status == "closed"
    assert closed[0].kind == "AccountClosed"

    assert projection_lag(7, 3) == 4
    assert unseen_event_ids((opened[0], opened[0], deposited[0]), {opened[0].event_id}) == (
        deposited[0].event_id,
    )
    _expect_error(lambda: projection_lag(1, 2), ValidationError)
    _expect_error(lambda: fold_balance((opened[0],)), ValidationError)

    rebuilt = AccountProjection()
    rebuilt.rebuild(store)
    assert rebuilt.get("acct-42") == projection.get("acct-42")
    _expect_error(
        lambda: AccountProjection().apply(store.stream("acct-42")[1]),
        ProjectionGap,
    )
    _expect_error(lambda: commands.close("acct-42", command_id="close-again"), AccountStateError)


def main() -> None:
    """Demonstrate a stale read, catch-up, and replayable read model."""
    run_self_checks()

    store = EventStore()
    commands = AccountCommandService(store)
    projection = AccountProjection()

    commands.open_account("demo-42", "Grace Hopper", command_id="demo-open")
    print("Immediately after command:", projection.get("demo-42"))
    projection.catch_up(store)
    commands.deposit("demo-42", 4_200, command_id="demo-deposit")
    print("Before projection catch-up:", projection.get("demo-42"))
    projection.catch_up(store)
    print("After projection catch-up:", projection.get("demo-42"))
    print("Produced events:", len(store.all_events()))
    print("Projection checkpoint:", projection.checkpoint)
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
