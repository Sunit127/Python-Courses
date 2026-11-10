"""Day 46: Workflow versioning, deterministic replay, and safe deployments.

Learning goals
--------------
1. Pin a long-running workflow to a version so a deploy cannot silently change history.
2. Keep replay decisions deterministic by injecting time, randomness, and external facts.
3. Upcast old event schemas at the boundary instead of branching everywhere in business code.
4. Use compatibility checks, canary gates, and rollback plans before activating a new definition.
5. Test replay equivalence and prove that a version change is intentional.

Teaching notes
--------------
- A workflow may run for hours, days, or months. Its history was produced by the
  definition that was deployed at that moment. Replaying old history with today's
  code is safe only when the decision semantics are unchanged.
- Store a definition version with every workflow instance. New executions can use
  version 2 while existing executions finish on version 1, or a deliberate
  migration can emit an explicit VersionUpgraded event.
- Replay should be a pure fold over normalized events. Do not read wall-clock time,
  call a service, generate a UUID, or inspect environment variables inside the fold.
  Record those facts as events or pass deterministic inputs to the decision function.
- Upcasting translates a historical event (for example, amount_dollars) into the
  current shape (amount_cents) once, at the boundary. The domain model then has one
  schema to reason about.
- A safe deployment checks that every pinned workflow version still has a compatible
  decoder, runs a replay corpus, and exposes the new definition to a small canary
  cohort. Rollback stops new assignments; it does not rewrite committed history.
- Versioning is not a substitute for idempotency. Commands can be delivered twice
  during a retry, so command IDs and participant deduplication remain mandatory.

Run this file with Python 3.10+.
It uses only the standard library, performs no network calls, and keeps state in memory.

Practice exercises
------------------
1. Add a v1 "ShipmentCreated" upcaster that supplies a default carrier.
2. Implement replay_digest(events, version) using canonical JSON and SHA-256.
3. Extend DeploymentGate to reject a version when its replay corpus has a mismatch.

Solutions appear in the executable code below.

Expert challenge: version-aware release controller
---------------------------------------------------
Build a release controller for the Day 45 process manager. Persist workflow
definitions, event upcasters, replay-corpus hashes, and canary assignments in SQLite.
Implement a two-phase rollout (shadow replay, then canary), per-tenant percentage
gates, automatic rollback on replay divergence or error-rate SLO breaches, and a
manual approval record. Keep old workers able to decode new events during the
rollback window. Add property-based tests for upcaster idempotence, deterministic
replay, duplicate command delivery, and crash points before and after each commit.
Document the compatibility matrix, migration procedure, rollback limits, and the
operator evidence required to declare a release safe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json


class WorkflowError(RuntimeError):
    """Base class for workflow-versioning failures."""


class ValidationError(WorkflowError, ValueError):
    """Raised when an input violates the workflow contract."""


class ReplayMismatch(WorkflowError):
    """Raised when replay does not match an expected deterministic result."""


class IncompatibleVersion(WorkflowError):
    """Raised when a deployment cannot safely serve a workflow version."""


class Status(Enum):
    NEW = "new"
    PAYMENT_PENDING = "payment_pending"
    READY_TO_SHIP = "ready_to_ship"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Event:
    """Historical event as stored by the workflow log."""

    kind: str
    data: Mapping[str, object]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValidationError("event kind must be non-empty text")
        if not isinstance(self.data, Mapping):
            raise ValidationError("event data must be a mapping")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValidationError("schema_version must be a positive integer")
        copied = dict(self.data)
        try:
            json.dumps(copied, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValidationError("event data must be JSON-compatible") from error
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "data", copied)


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """State produced solely by folding normalized events."""

    workflow_id: str
    version: str
    status: Status = Status.NEW
    amount_cents: int = 0
    payment_id: str = ""
    shipment_id: str = ""
    last_event_number: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.workflow_id, str) or not self.workflow_id.strip():
            raise ValidationError("workflow_id must be non-empty text")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValidationError("version must be non-empty text")
        if self.amount_cents < 0:
            raise ValidationError("amount_cents cannot be negative")


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be non-empty text")
    return value.strip()


def _positive_cents(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("amount_cents must be a positive integer")
    return value


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise ValidationError("value is not canonical JSON") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def upcast(event: Event) -> Event:
    """Translate every supported historical shape into schema version 2."""
    if not isinstance(event, Event):
        raise ValidationError("event must be an Event")
    if event.schema_version == 2:
        return event
    if event.schema_version != 1:
        raise IncompatibleVersion(f"unsupported event schema {event.schema_version}")

    data = dict(event.data)
    if event.kind == "OrderStarted":
        if "amount_cents" not in data and "amount_dollars" in data:
            dollars = data.pop("amount_dollars")
            if (
                isinstance(dollars, bool)
                or not isinstance(dollars, (int, float))
                or dollars <= 0
            ):
                raise ValidationError("amount_dollars must be a positive number")
            cents = round(float(dollars) * 100)
            if cents <= 0:
                raise ValidationError("amount_dollars rounds to zero cents")
            data["amount_cents"] = cents
        data.setdefault("currency", "USD")
    return Event(event.kind, data, schema_version=2)


def normalize_events(events: Iterable[Event]) -> tuple[Event, ...]:
    """Upcast once at the replay boundary and reject malformed streams."""
    try:
        items = tuple(events)
    except TypeError as error:
        raise ValidationError("events must be iterable") from error
    if any(not isinstance(item, Event) for item in items):
        raise ValidationError("events must contain Event values")
    return tuple(upcast(item) for item in items)


def _event_data(event: Event, key: str) -> object:
    try:
        return event.data[key]
    except KeyError as error:
        raise ValidationError(f"{event.kind} requires {key}") from error


def apply_event(state: WorkflowState, event: Event, event_number: int) -> WorkflowState:
    """Pure state transition: same state and event always yield the same result."""
    if not isinstance(state, WorkflowState) or not isinstance(event, Event):
        raise ValidationError("state and event must have the expected types")
    if event_number != state.last_event_number + 1:
        raise ReplayMismatch("event numbers must be contiguous")
    event = upcast(event)
    data = event.data

    if event.kind == "OrderStarted":
        if state.status is not Status.NEW:
            raise ReplayMismatch("order can only start from NEW")
        amount = _positive_cents(data.get("amount_cents"))
        return replace(
            state,
            amount_cents=amount,
            status=Status.PAYMENT_PENDING,
            last_event_number=event_number,
        )

    if event.kind == "PaymentAuthorized":
        if state.status is not Status.PAYMENT_PENDING:
            raise ReplayMismatch("payment is not expected in the current state")
        payment_id = _text(_event_data(event, "payment_id"), field="payment_id")
        return replace(
            state,
            payment_id=payment_id,
            status=Status.READY_TO_SHIP,
            last_event_number=event_number,
        )

    if event.kind == "ShipmentCreated":
        if state.status is not Status.READY_TO_SHIP:
            raise ReplayMismatch("shipment is not expected in the current state")
        shipment_id = _text(_event_data(event, "shipment_id"), field="shipment_id")
        return replace(
            state,
            shipment_id=shipment_id,
            status=Status.COMPLETED,
            last_event_number=event_number,
        )

    if event.kind == "OrderFailed":
        if state.status is Status.COMPLETED:
            raise ReplayMismatch("a completed order cannot fail")
        return replace(state, status=Status.FAILED, last_event_number=event_number)

    raise IncompatibleVersion(f"unknown event kind: {event.kind}")


def replay(
    workflow_id: str, version: str, events: Iterable[Event]
) -> WorkflowState:
    """Replay a history using a pinned definition version."""
    clean_id = _text(workflow_id, field="workflow_id")
    clean_version = _text(version, field="version")
    state = WorkflowState(clean_id, clean_version)
    for number, event in enumerate(normalize_events(events), start=1):
        state = apply_event(state, event, number)
    return state


@dataclass(frozen=True, slots=True)
class Decision:
    """A deterministic command decision made from a state and a recorded fact."""

    command_name: str
    command_id: str
    payload: Mapping[str, object]


def decide_next(state: WorkflowState, *, fact: Mapping[str, object]) -> Decision | None:
    """Return the next command without reading time, randomness, or the network."""
    if not isinstance(state, WorkflowState) or not isinstance(fact, Mapping):
        raise ValidationError("state and fact have invalid types")
    if state.status is Status.PAYMENT_PENDING:
        request_id = _text(fact.get("request_id"), field="request_id")
        return Decision(
            "authorize_payment",
            f"{state.workflow_id}:authorize:{request_id}",
            {"amount_cents": state.amount_cents},
        )
    if state.status is Status.READY_TO_SHIP:
        request_id = _text(fact.get("request_id"), field="request_id")
        return Decision(
            "create_shipment",
            f"{state.workflow_id}:ship:{request_id}",
            {"payment_id": state.payment_id},
        )
    return None


class ReleasePhase(Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    version: str
    phase: ReleasePhase
    replayed: int
    mismatches: tuple[str, ...]


class DeploymentGate:
    """Small, deterministic release gate for versioned workflow definitions."""

    def __init__(self, supported_versions: Iterable[str]) -> None:
        versions = tuple(_text(version, field="version") for version in supported_versions)
        if not versions or len(set(versions)) != len(versions):
            raise ValidationError("supported_versions must be unique and non-empty")
        self._supported = frozenset(versions)

    def evaluate(
        self,
        version: str,
        corpus: Sequence[tuple[str, Sequence[Event], str]],
        *,
        phase: ReleasePhase,
    ) -> ReleaseReport:
        clean_version = _text(version, field="version")
        if clean_version not in self._supported:
            raise IncompatibleVersion(f"{clean_version!r} is not supported")
        if not isinstance(phase, ReleasePhase):
            raise ValidationError("phase must be a ReleasePhase")
        mismatches: list[str] = []
        replayed = 0
        for workflow_id, events, expected_digest in corpus:
            try:
                state = replay(workflow_id, clean_version, events)
                actual = _digest({
                    "workflow_id": state.workflow_id,
                    "version": state.version,
                    "status": state.status.value,
                    "amount_cents": state.amount_cents,
                    "payment_id": state.payment_id,
                    "shipment_id": state.shipment_id,
                    "last_event_number": state.last_event_number,
                })
                replayed += 1
                if actual != expected_digest:
                    mismatches.append(workflow_id)
            except WorkflowError:
                mismatches.append(workflow_id)
        if mismatches and phase in {ReleasePhase.CANARY, ReleasePhase.ACTIVE}:
            phase = ReleasePhase.BLOCKED
        return ReleaseReport(clean_version, phase, replayed, tuple(mismatches))


def _expect_error(callback: Callable[[], object], expected: type[BaseException]) -> None:
    try:
        callback()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def _state_digest(state: WorkflowState) -> str:
    return _digest({
        "workflow_id": state.workflow_id,
        "version": state.version,
        "status": state.status.value,
        "amount_cents": state.amount_cents,
        "payment_id": state.payment_id,
        "shipment_id": state.shipment_id,
        "last_event_number": state.last_event_number,
    })


def run_self_checks() -> None:
    """Check upcasting, deterministic replay, command IDs, and release gates."""
    old_history = (
        Event("OrderStarted", {"amount_dollars": 12.50}),
        Event("PaymentAuthorized", {"payment_id": "pay-7"}),
        Event("ShipmentCreated", {"shipment_id": "ship-7", "carrier": "Acme"}),
    )
    state = replay("order-7", "v1", old_history)
    assert state.status is Status.COMPLETED
    assert state.amount_cents == 1250
    assert normalize_events(old_history)[0].schema_version == 2

    repeated = replay("order-7", "v1", old_history)
    assert repeated == state
    assert decide_next(
        replay("order-8", "v1", (Event("OrderStarted", {"amount_cents": 900}),)),
        fact={"request_id": "req-1"},
    ) == Decision("authorize_payment", "order-8:authorize:req-1", {"amount_cents": 900})

    _expect_error(
        lambda: replay(
            "order-9", "v1",
            (Event("OrderStarted", {"amount_cents": 1}), Event("ShipmentCreated", {"shipment_id": "x"})),
        ),
        ReplayMismatch,
    )
    _expect_error(
        lambda: replay("order-10", "v1", (Event("Unknown", {}),)),
        IncompatibleVersion,
    )

    expected_v2 = _state_digest(replay("order-7", "v2", old_history))
    corpus = (("order-7", old_history, expected_v2),)
    report = DeploymentGate({"v1", "v2"}).evaluate(
        "v2", corpus, phase=ReleasePhase.CANARY
    )
    assert report.replayed == 1 and report.mismatches == ()
    assert report.phase is ReleasePhase.CANARY

    mismatch = DeploymentGate({"v1"}).evaluate(
        "v1", corpus, phase=ReleasePhase.ACTIVE
    )
    assert mismatch.mismatches == ("order-7",)
    assert mismatch.phase is ReleasePhase.BLOCKED
    print("Self-checks passed.")


def main() -> None:
    """Demonstrate safe replay and a canary gate."""
    run_self_checks()
    history = (
        Event("OrderStarted", {"amount_dollars": 19.99}),
        Event("PaymentAuthorized", {"payment_id": "pay-demo"}),
    )
    state = replay("demo-order", "v2", history)
    print("Pinned version:", state.version)
    print("Status:", state.status.value)
    print("Amount (cents):", state.amount_cents)
    print("Next decision:", decide_next(state, fact={"request_id": "req-demo"}))
    print("Replay digest:", _state_digest(state)[:16] + "...")


if __name__ == "__main__":
    main()
