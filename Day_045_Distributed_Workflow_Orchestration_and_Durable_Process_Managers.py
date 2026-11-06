"""Day 45: Distributed workflow orchestration and durable process managers.

Learning goals
--------------
1. Model a long-running business process as state rebuilt from durable history.
2. Separate deterministic workflow decisions from side-effecting workers.
3. Persist commands and timers with workflow events so crashes do not lose work.
4. Handle duplicate signals, optimistic-concurrency conflicts, and retries safely.
5. Explain why orchestration still requires idempotent downstream participants.

Teaching notes
--------------
- A process manager coordinates work that spans services and time. Unlike a
  request handler, it must survive restarts, delayed replies, and duplicate
  delivery without keeping correctness-critical state only in memory.
- The workflow history is the source of truth. Replaying WorkflowStarted,
  StepRequested, StepSucceeded, StepFailed, and retry events reconstructs the
  same state. Decision code should depend only on that state and the new signal.
- Persist an outgoing command in the same commit as the event that requested
  it. This is a workflow outbox. A dispatcher may send a command more than once
  after a crash, so the participant must deduplicate the stable command_id.
- Timers are durable messages, not sleeps. A timer scanner can rediscover a due
  timer after restart. Its stable signal id makes repeated delivery harmless.
- Optimistic concurrency rejects two coordinators that decide from the same
  history version. A production engine reloads and retries the losing decision.
- Signal deduplication needs both an id and a fingerprint. Replaying the same
  signal is safe; reusing its id for different data is a correctness error.
- Orchestration makes the control flow visible, but it is not a distributed
  transaction. Completed remote effects may need compensation, reconciliation,
  or manual review, as introduced in Day 41.
- Keep workflow code deterministic. Wall-clock reads, random ids, network calls,
  and environment access belong outside the decision function and enter through
  recorded signals, commands, or timers.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, and keeps all
state in memory for teaching purposes.

Practice exercises
------------------
1. Implement exponential_backoff(attempt, base_seconds, maximum_seconds).
2. Implement outstanding_commands(commands, dispatched_ids) while preserving
   command order and rejecting malformed input.
3. Implement workflow_summary(history) by replaying history rather than reading
   mutable fields from the store.

Solutions appear in the executable code below.

Expert challenge: production workflow control plane
----------------------------------------------------
Build a durable order-fulfilment engine backed by a transactional database.
Store history, signal deduplication records, command outbox entries, and timers
atomically. Add worker leases, command heartbeats, bounded retries, dead-letter
handling, compensation, cancellation, per-tenant limits, encrypted payloads,
and operator pause/resume controls. Test crash points before and after every
commit and dispatch. Add deterministic replay tests, history schema upcasters,
workflow-version pinning, metrics for queue age and stuck workflows, distributed
tracing, audit exports, and a migration/rollback runbook. Document delivery
semantics and every place that still requires idempotency or reconciliation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math


class WorkflowError(RuntimeError):
    """Base class for workflow coordination failures."""


class ValidationError(WorkflowError, ValueError):
    """Raised when an external value violates the workflow contract."""


class ConcurrentUpdate(WorkflowError):
    """Raised when a coordinator commits from a stale history version."""


class SignalConflict(WorkflowError):
    """Raised when a signal id is reused with different content."""


class InvalidTransition(WorkflowError):
    """Raised when a signal is impossible for the current state."""


class Status(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


STEPS = ("authorize_payment", "reserve_inventory", "create_shipment")


def _text(value: object, *, field: str, limit: int = 120) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValidationError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise ValidationError(f"{field} must be at most {limit} characters")
    return cleaned


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{field} must be finite")
    return result


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be an object")
    copied = dict(value)
    if any(not isinstance(key, str) for key in copied):
        raise ValidationError(f"{field} keys must be text")
    try:
        json.dumps(copied, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be JSON-compatible") from error
    return copied


def _fingerprint(kind: str, data: Mapping[str, object]) -> str:
    encoded = json.dumps(
        {"kind": kind, "data": dict(data)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Signal:
    signal_id: str
    kind: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_id", _text(self.signal_id, field="signal_id"))
        object.__setattr__(self, "kind", _text(self.kind, field="kind"))
        object.__setattr__(self, "data", _json_object(self.data, field="data"))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.kind, self.data)


@dataclass(frozen=True, slots=True)
class PendingEvent:
    kind: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, field="event kind"))
        object.__setattr__(self, "data", _json_object(self.data, field="event data"))


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    sequence: int
    kind: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", _positive_int(self.sequence, field="sequence")
        )
        object.__setattr__(self, "kind", _text(self.kind, field="event kind"))
        object.__setattr__(self, "data", _json_object(self.data, field="event data"))


@dataclass(frozen=True, slots=True)
class Command:
    command_id: str
    workflow_id: str
    name: str
    attempt: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _text(self.command_id, field="command_id")
        )
        object.__setattr__(
            self, "workflow_id", _text(self.workflow_id, field="workflow_id")
        )
        object.__setattr__(self, "name", _text(self.name, field="command name"))
        object.__setattr__(
            self, "attempt", _positive_int(self.attempt, field="attempt")
        )
        object.__setattr__(
            self, "payload", _json_object(self.payload, field="command payload")
        )


@dataclass(frozen=True, slots=True)
class Timer:
    timer_id: str
    workflow_id: str
    due_at: float
    signal: Signal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timer_id", _text(self.timer_id, field="timer_id"))
        object.__setattr__(
            self, "workflow_id", _text(self.workflow_id, field="workflow_id")
        )
        object.__setattr__(
            self, "due_at", _finite(self.due_at, field="due_at")
        )
        if not isinstance(self.signal, Signal):
            raise ValidationError("timer signal must be a Signal")


@dataclass(frozen=True, slots=True)
class Decision:
    events: tuple[PendingEvent, ...] = ()
    commands: tuple[Command, ...] = ()
    timers: tuple[Timer, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessState:
    workflow_id: str
    version: int = 0
    status: Status = Status.NOT_STARTED
    order_id: str | None = None
    amount_cents: int | None = None
    completed_steps: tuple[str, ...] = ()
    pending_step: str | None = None
    pending_attempt: int | None = None
    retry_step: str | None = None
    retry_attempt: int | None = None
    retry_due_at: float | None = None
    failure: str | None = None


def evolve(state: ProcessState, event: WorkflowEvent) -> ProcessState:
    """Apply one recorded event while enforcing sequence and state invariants."""
    if not isinstance(state, ProcessState) or not isinstance(event, WorkflowEvent):
        raise ValidationError("evolve requires ProcessState and WorkflowEvent")
    if event.sequence != state.version + 1:
        raise InvalidTransition("history sequence is not contiguous")

    next_state = replace(state, version=event.sequence)
    data = event.data

    if event.kind == "WorkflowStarted":
        if state.status is not Status.NOT_STARTED:
            raise InvalidTransition("workflow can start only once")
        order_id = _text(data.get("order_id"), field="order_id")
        amount = _positive_int(data.get("amount_cents"), field="amount_cents")
        return replace(
            next_state,
            status=Status.RUNNING,
            order_id=order_id,
            amount_cents=amount,
        )

    if event.kind == "StepRequested":
        if state.status is not Status.RUNNING or state.pending_step is not None:
            raise InvalidTransition("a step cannot be requested now")
        step = _text(data.get("step"), field="step")
        if step not in STEPS:
            raise InvalidTransition(f"unknown workflow step: {step}")
        attempt = _positive_int(data.get("attempt"), field="attempt")
        return replace(
            next_state, pending_step=step, pending_attempt=attempt, failure=None
        )

    if event.kind == "StepSucceeded":
        step = _text(data.get("step"), field="step")
        attempt = _positive_int(data.get("attempt"), field="attempt")
        if (step, attempt) != (state.pending_step, state.pending_attempt):
            raise InvalidTransition("success does not match the pending command")
        if step in state.completed_steps:
            raise InvalidTransition("step is already complete")
        return replace(
            next_state,
            completed_steps=state.completed_steps + (step,),
            pending_step=None,
            pending_attempt=None,
        )

    if event.kind == "StepFailed":
        step = _text(data.get("step"), field="step")
        attempt = _positive_int(data.get("attempt"), field="attempt")
        reason = _text(data.get("reason"), field="reason", limit=240)
        if (step, attempt) != (state.pending_step, state.pending_attempt):
            raise InvalidTransition("failure does not match the pending command")
        return replace(
            next_state,
            pending_step=None,
            pending_attempt=None,
            failure=reason,
        )

    if event.kind == "RetryScheduled":
        if state.status is not Status.RUNNING or state.pending_step is not None:
            raise InvalidTransition("retry cannot be scheduled now")
        step = _text(data.get("step"), field="step")
        attempt = _positive_int(data.get("attempt"), field="attempt")
        due_at = _finite(data.get("due_at"), field="due_at")
        return replace(
            next_state,
            status=Status.WAITING,
            retry_step=step,
            retry_attempt=attempt,
            retry_due_at=due_at,
        )

    if event.kind == "RetryTimerFired":
        timer_id = _text(data.get("timer_id"), field="timer_id")
        if not timer_id or state.status is not Status.WAITING:
            raise InvalidTransition("retry timer is not expected")
        return replace(
            next_state,
            status=Status.RUNNING,
            retry_step=None,
            retry_attempt=None,
            retry_due_at=None,
        )

    if event.kind == "WorkflowCompleted":
        if state.pending_step is not None or state.completed_steps != STEPS:
            raise InvalidTransition("workflow cannot complete before every step")
        return replace(next_state, status=Status.COMPLETED, failure=None)

    if event.kind == "WorkflowFailed":
        reason = _text(data.get("reason"), field="reason", limit=240)
        if state.pending_step is not None:
            raise InvalidTransition("record the step failure before workflow failure")
        return replace(next_state, status=Status.FAILED, failure=reason)

    raise InvalidTransition(f"unsupported event kind: {event.kind}")


def replay(workflow_id: str, history: Iterable[WorkflowEvent]) -> ProcessState:
    """Rebuild process state solely from ordered history."""
    clean_id = _text(workflow_id, field="workflow_id")
    try:
        events = tuple(history)
    except TypeError as error:
        raise ValidationError("history must be iterable") from error
    state = ProcessState(clean_id)
    for event in events:
        if not isinstance(event, WorkflowEvent):
            raise ValidationError("history must contain WorkflowEvent values")
        state = evolve(state, event)
    return state


def exponential_backoff(
    attempt: int,
    base_seconds: float = 2.0,
    maximum_seconds: float = 60.0,
) -> float:
    """Solution 1: bounded exponential delay for a one-based attempt."""
    clean_attempt = _positive_int(attempt, field="attempt")
    base = _finite(base_seconds, field="base_seconds")
    maximum = _finite(maximum_seconds, field="maximum_seconds")
    if base <= 0 or maximum <= 0:
        raise ValidationError("backoff durations must be positive")
    return min(maximum, base * (2 ** (clean_attempt - 1)))


def outstanding_commands(
    commands: Iterable[Command], dispatched_ids: Iterable[str]
) -> tuple[Command, ...]:
    """Solution 2: preserve order and omit commands marked as dispatched."""
    try:
        items = tuple(commands)
        dispatched = frozenset(dispatched_ids)
    except TypeError as error:
        raise ValidationError("commands and dispatched_ids must be iterable") from error
    if any(not isinstance(command, Command) for command in items):
        raise ValidationError("commands must contain Command values")
    if any(not isinstance(item, str) or not item for item in dispatched):
        raise ValidationError("dispatched ids must be non-empty text")
    seen: set[str] = set()
    result: list[Command] = []
    for command in items:
        if command.command_id in seen:
            raise ValidationError("command ids must be unique")
        seen.add(command.command_id)
        if command.command_id not in dispatched:
            result.append(command)
    return tuple(result)


def workflow_summary(history: Iterable[WorkflowEvent]) -> dict[str, object]:
    """Solution 3: derive an operator summary by replaying the history."""
    try:
        events = tuple(history)
    except TypeError as error:
        raise ValidationError("history must be iterable") from error
    if not events:
        raise ValidationError("history must not be empty")
    first = events[0]
    if not isinstance(first, WorkflowEvent):
        raise ValidationError("history must contain WorkflowEvent values")
    workflow_id = first.data.get("workflow_id")
    clean_id = _text(workflow_id, field="workflow_id")
    state = replay(clean_id, events)
    return {
        "workflow_id": state.workflow_id,
        "version": state.version,
        "status": state.status.value,
        "completed_steps": state.completed_steps,
        "pending_step": state.pending_step,
        "failure": state.failure,
    }


class ProcessManager:
    """Pure decision logic for the order-fulfilment workflow."""

    def __init__(self, *, max_attempts: int = 3) -> None:
        self.max_attempts = _positive_int(max_attempts, field="max_attempts")

    @staticmethod
    def _command(
        state: ProcessState, step: str, attempt: int
    ) -> tuple[PendingEvent, Command]:
        if state.order_id is None or state.amount_cents is None:
            raise InvalidTransition("workflow has no order context")
        command_id = f"{state.workflow_id}:{step}:{attempt}"
        event = PendingEvent(
            "StepRequested",
            {"step": step, "attempt": attempt},
        )
        command = Command(
            command_id,
            state.workflow_id,
            step,
            attempt,
            {
                "order_id": state.order_id,
                "amount_cents": state.amount_cents,
            },
        )
        return event, command

    def decide(
        self, state: ProcessState, signal: Signal, *, now: float
    ) -> Decision:
        if not isinstance(state, ProcessState) or not isinstance(signal, Signal):
            raise ValidationError("decide requires ProcessState and Signal")
        current_time = _finite(now, field="now")

        if signal.kind == "Start":
            if state.status is not Status.NOT_STARTED:
                raise InvalidTransition("workflow already started")
            order_id = _text(signal.data.get("order_id"), field="order_id")
            amount = _positive_int(
                signal.data.get("amount_cents"), field="amount_cents"
            )
            started = PendingEvent(
                "WorkflowStarted",
                {
                    "workflow_id": state.workflow_id,
                    "order_id": order_id,
                    "amount_cents": amount,
                },
            )
            started_state = evolve(
                state, WorkflowEvent(state.version + 1, started.kind, started.data)
            )
            requested, command = self._command(
                started_state, STEPS[0], attempt=1
            )
            return Decision((started, requested), (command,))

        if signal.kind == "StepSucceeded":
            if state.status is not Status.RUNNING:
                raise InvalidTransition("workflow is not accepting step results")
            step = _text(signal.data.get("step"), field="step")
            attempt = _positive_int(signal.data.get("attempt"), field="attempt")
            if (step, attempt) != (state.pending_step, state.pending_attempt):
                raise InvalidTransition("success does not match pending work")
            succeeded = PendingEvent(
                "StepSucceeded", {"step": step, "attempt": attempt}
            )
            after_success = evolve(
                state,
                WorkflowEvent(state.version + 1, succeeded.kind, succeeded.data),
            )
            index = STEPS.index(step)
            if index == len(STEPS) - 1:
                return Decision(
                    (succeeded, PendingEvent("WorkflowCompleted", {}))
                )
            requested, command = self._command(
                after_success, STEPS[index + 1], attempt=1
            )
            return Decision((succeeded, requested), (command,))

        if signal.kind == "StepFailed":
            if state.status is not Status.RUNNING:
                raise InvalidTransition("workflow is not accepting step results")
            step = _text(signal.data.get("step"), field="step")
            attempt = _positive_int(signal.data.get("attempt"), field="attempt")
            transient = signal.data.get("transient")
            if not isinstance(transient, bool):
                raise ValidationError("transient must be boolean")
            reason = _text(signal.data.get("reason"), field="reason", limit=240)
            if (step, attempt) != (state.pending_step, state.pending_attempt):
                raise InvalidTransition("failure does not match pending work")
            failed = PendingEvent(
                "StepFailed",
                {"step": step, "attempt": attempt, "reason": reason},
            )
            if not transient or attempt >= self.max_attempts:
                return Decision(
                    (failed, PendingEvent("WorkflowFailed", {"reason": reason}))
                )

            next_attempt = attempt + 1
            due_at = current_time + exponential_backoff(attempt)
            timer_id = f"{state.workflow_id}:{step}:retry:{next_attempt}"
            timer_signal = Signal(
                f"timer:{timer_id}",
                "RetryDue",
                {
                    "timer_id": timer_id,
                    "step": step,
                    "attempt": next_attempt,
                    "due_at": due_at,
                },
            )
            timer = Timer(timer_id, state.workflow_id, due_at, timer_signal)
            scheduled = PendingEvent(
                "RetryScheduled",
                {"step": step, "attempt": next_attempt, "due_at": due_at},
            )
            return Decision((failed, scheduled), timers=(timer,))

        if signal.kind == "RetryDue":
            if state.status is not Status.WAITING:
                raise InvalidTransition("workflow is not waiting for a retry")
            timer_id = _text(signal.data.get("timer_id"), field="timer_id")
            step = _text(signal.data.get("step"), field="step")
            attempt = _positive_int(signal.data.get("attempt"), field="attempt")
            due_at = _finite(signal.data.get("due_at"), field="due_at")
            if (
                step != state.retry_step
                or attempt != state.retry_attempt
                or due_at != state.retry_due_at
            ):
                raise InvalidTransition("timer does not match scheduled retry")
            if current_time < due_at:
                raise InvalidTransition("retry timer is not due")
            fired = PendingEvent("RetryTimerFired", {"timer_id": timer_id})
            running = evolve(
                state, WorkflowEvent(state.version + 1, fired.kind, fired.data)
            )
            requested, command = self._command(running, step, attempt)
            return Decision((fired, requested), (command,))

        raise InvalidTransition(f"unsupported signal kind: {signal.kind}")


@dataclass(frozen=True, slots=True)
class CommitResult:
    applied: bool
    new_version: int


class MemoryWorkflowStore:
    """Atomic teaching store for history, deduplication, outbox, and timers."""

    def __init__(self) -> None:
        self._histories: dict[str, list[WorkflowEvent]] = {}
        self._signals: dict[tuple[str, str], str] = {}
        self._commands: dict[str, Command] = {}
        self._command_order: list[str] = []
        self._dispatched: set[str] = set()
        self._timers: dict[str, Timer] = {}

    def history(self, workflow_id: str) -> tuple[WorkflowEvent, ...]:
        clean_id = _text(workflow_id, field="workflow_id")
        return tuple(self._histories.get(clean_id, ()))

    def check_signal(
        self, workflow_id: str, signal: Signal
    ) -> CommitResult | None:
        """Return a no-op result for a known signal before making a decision."""
        clean_id = _text(workflow_id, field="workflow_id")
        if not isinstance(signal, Signal):
            raise ValidationError("signal must be a Signal")
        previous = self._signals.get((clean_id, signal.signal_id))
        if previous is None:
            return None
        if previous != signal.fingerprint:
            raise SignalConflict(
                f"signal id {signal.signal_id!r} has conflicting content"
            )
        return CommitResult(False, len(self._histories.get(clean_id, ())))

    def commit(
        self,
        workflow_id: str,
        *,
        expected_version: int,
        signal: Signal,
        decision: Decision,
    ) -> CommitResult:
        clean_id = _text(workflow_id, field="workflow_id")
        if isinstance(expected_version, bool) or not isinstance(expected_version, int):
            raise ValidationError("expected_version must be an integer")
        if expected_version < 0:
            raise ValidationError("expected_version must not be negative")
        if not isinstance(signal, Signal) or not isinstance(decision, Decision):
            raise ValidationError("commit requires Signal and Decision")

        signal_key = (clean_id, signal.signal_id)
        previous = self._signals.get(signal_key)
        if previous is not None:
            if previous != signal.fingerprint:
                raise SignalConflict(
                    f"signal id {signal.signal_id!r} has conflicting content"
                )
            return CommitResult(False, len(self._histories.get(clean_id, ())))

        history = self._histories.setdefault(clean_id, [])
        if len(history) != expected_version:
            raise ConcurrentUpdate(
                f"expected version {expected_version}, found {len(history)}"
            )

        for command in decision.commands:
            if command.workflow_id != clean_id:
                raise ValidationError("command belongs to another workflow")
            existing = self._commands.get(command.command_id)
            if existing is not None and existing != command:
                raise WorkflowError("command id has conflicting content")
        for timer in decision.timers:
            if timer.workflow_id != clean_id:
                raise ValidationError("timer belongs to another workflow")
            existing_timer = self._timers.get(timer.timer_id)
            if existing_timer is not None and existing_timer != timer:
                raise WorkflowError("timer id has conflicting content")

        start = len(history)
        recorded = [
            WorkflowEvent(start + offset, event.kind, event.data)
            for offset, event in enumerate(decision.events, start=1)
        ]

        # One logical transaction begins here in the teaching store.
        history.extend(recorded)
        self._signals[signal_key] = signal.fingerprint
        for command in decision.commands:
            if command.command_id not in self._commands:
                self._commands[command.command_id] = command
                self._command_order.append(command.command_id)
        for timer in decision.timers:
            self._timers.setdefault(timer.timer_id, timer)
        return CommitResult(True, len(history))

    def pending_commands(self) -> tuple[Command, ...]:
        ordered = (self._commands[item] for item in self._command_order)
        return outstanding_commands(ordered, self._dispatched)

    def mark_dispatched(self, command_id: str) -> None:
        clean_id = _text(command_id, field="command_id")
        if clean_id not in self._commands:
            raise ValidationError("unknown command_id")
        self._dispatched.add(clean_id)

    def due_timers(self, now: float) -> tuple[Timer, ...]:
        current = _finite(now, field="now")
        return tuple(
            sorted(
                (timer for timer in self._timers.values() if timer.due_at <= current),
                key=lambda timer: (timer.due_at, timer.timer_id),
            )
        )


class ManualClock:
    def __init__(self, now: float = 0.0) -> None:
        self._now = _finite(now, field="now")

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        amount = _finite(seconds, field="seconds")
        if amount < 0:
            raise ValidationError("clock cannot move backwards")
        self._now += amount


class WorkflowEngine:
    """Replay, decide, and atomically commit one signal."""

    def __init__(
        self,
        store: MemoryWorkflowStore,
        manager: ProcessManager,
        clock: ManualClock,
    ) -> None:
        self.store = store
        self.manager = manager
        self.clock = clock

    def handle(self, workflow_id: str, signal: Signal) -> CommitResult:
        duplicate = self.store.check_signal(workflow_id, signal)
        if duplicate is not None:
            return duplicate
        history = self.store.history(workflow_id)
        state = replay(workflow_id, history)
        decision = self.manager.decide(state, signal, now=self.clock.now())
        return self.store.commit(
            workflow_id,
            expected_version=state.version,
            signal=signal,
            decision=decision,
        )

    def fire_due_timers(self) -> tuple[CommitResult, ...]:
        results: list[CommitResult] = []
        for timer in self.store.due_timers(self.clock.now()):
            results.append(self.handle(timer.workflow_id, timer.signal))
        return tuple(results)


def _expect_error(callback: object, expected: type[BaseException]) -> None:
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        callback()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def _success(command: Command, signal_id: str) -> Signal:
    return Signal(
        signal_id,
        "StepSucceeded",
        {"step": command.name, "attempt": command.attempt},
    )


def run_self_checks() -> None:
    """Exercise success, retry, replay, deduplication, and conflict paths."""
    clock = ManualClock(100.0)
    store = MemoryWorkflowStore()
    engine = WorkflowEngine(store, ProcessManager(max_attempts=3), clock)
    workflow_id = "order-workflow-45"

    start = Signal(
        "start-45",
        "Start",
        {"order_id": "order-45", "amount_cents": 12_500},
    )
    result = engine.handle(workflow_id, start)
    assert result.applied and result.new_version == 2

    # An exact duplicate is harmless at the storage boundary.
    duplicate_decision = Decision()
    duplicate = store.commit(
        workflow_id,
        expected_version=0,
        signal=start,
        decision=duplicate_decision,
    )
    assert not duplicate.applied and duplicate.new_version == 2
    _expect_error(
        lambda: store.commit(
            workflow_id,
            expected_version=2,
            signal=Signal(
                "start-45",
                "Start",
                {"order_id": "other", "amount_cents": 1},
            ),
            decision=Decision(),
        ),
        SignalConflict,
    )

    payment = store.pending_commands()[0]
    assert payment.name == "authorize_payment"
    store.mark_dispatched(payment.command_id)
    engine.handle(workflow_id, _success(payment, "payment-ok"))

    inventory = store.pending_commands()[0]
    store.mark_dispatched(inventory.command_id)
    failure = Signal(
        "inventory-timeout",
        "StepFailed",
        {
            "step": inventory.name,
            "attempt": inventory.attempt,
            "transient": True,
            "reason": "inventory service timed out",
        },
    )
    engine.handle(workflow_id, failure)
    waiting = replay(workflow_id, store.history(workflow_id))
    assert waiting.status is Status.WAITING
    assert waiting.retry_attempt == 2
    assert engine.fire_due_timers() == ()

    clock.advance(2.0)
    fired = engine.fire_due_timers()
    assert len(fired) == 1 and fired[0].applied
    # The durable timer remains discoverable, but stable signal dedup makes
    # another scan a no-op rather than creating a second command.
    fired_again = engine.fire_due_timers()
    assert len(fired_again) == 1 and not fired_again[0].applied

    inventory_retry = store.pending_commands()[0]
    assert inventory_retry.name == "reserve_inventory"
    assert inventory_retry.attempt == 2
    store.mark_dispatched(inventory_retry.command_id)
    engine.handle(
        workflow_id, _success(inventory_retry, "inventory-retry-ok")
    )

    shipment = store.pending_commands()[0]
    assert shipment.name == "create_shipment"
    store.mark_dispatched(shipment.command_id)
    engine.handle(workflow_id, _success(shipment, "shipment-ok"))

    history = store.history(workflow_id)
    final_state = replay(workflow_id, history)
    assert final_state.status is Status.COMPLETED
    assert final_state.completed_steps == STEPS
    assert store.pending_commands() == ()
    summary = workflow_summary(history)
    assert summary["status"] == "completed"
    assert summary["version"] == len(history)

    assert exponential_backoff(1) == 2.0
    assert exponential_backoff(4, maximum_seconds=10.0) == 10.0
    _expect_error(lambda: exponential_backoff(0), ValidationError)
    _expect_error(
        lambda: store.commit(
            "new-workflow",
            expected_version=1,
            signal=Signal("signal", "Start", {}),
            decision=Decision(),
        ),
        ConcurrentUpdate,
    )

    # A non-transient failure ends a separate workflow without a retry timer.
    failed_id = "failed-workflow-45"
    engine.handle(
        failed_id,
        Signal(
            "failed-start",
            "Start",
            {"order_id": "order-failed", "amount_cents": 500},
        ),
    )
    failed_command = store.pending_commands()[0]
    engine.handle(
        failed_id,
        Signal(
            "payment-declined",
            "StepFailed",
            {
                "step": failed_command.name,
                "attempt": failed_command.attempt,
                "transient": False,
                "reason": "payment declined",
            },
        ),
    )
    assert replay(failed_id, store.history(failed_id)).status is Status.FAILED


def main() -> None:
    run_self_checks()

    clock = ManualClock()
    store = MemoryWorkflowStore()
    engine = WorkflowEngine(store, ProcessManager(), clock)
    workflow_id = "demo-45"
    engine.handle(
        workflow_id,
        Signal(
            "demo-start",
            "Start",
            {"order_id": "demo-order", "amount_cents": 4_200},
        ),
    )

    first_command = store.pending_commands()[0]
    print("First durable command:", first_command)
    print("Replayed state:", replay(workflow_id, store.history(workflow_id)))
    print("History summary:", workflow_summary(store.history(workflow_id)))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
