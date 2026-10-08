"""Day 33: Reliable automation, idempotency, and resumable workflows.

Learning goals
--------------
1. Model automation as explicit, versioned steps instead of one fragile script.
2. Persist checkpoints with optimistic concurrency so interrupted work can resume.
3. Make side effects idempotent with stable keys and conflict detection.
4. Retry only classified transient failures with bounded attempts.
5. Test crash windows, replay, stale workers, and definition changes.

Teaching notes
--------------
- Reliable automation assumes interruption. A process may stop after a remote
  effect succeeds but before local progress is saved, so replay must be safe.
- A checkpoint records durable facts: the workflow identity, definition version,
  input fingerprint, next step, state, and revision. Do not store live objects,
  open files, sockets, or callbacks in it.
- Idempotency means the same logical request can be repeated without duplicating
  its effect. A reused key with different input is a conflict, not a cache hit.
- "Exactly once" is rarely available across independent systems. A practical
  design uses at-least-once delivery plus deduplication at the effect boundary.
- Save progress after a step succeeds. Saving before the effect risks silently
  skipping work; saving afterward creates a replay window that idempotency must
  close.
- Optimistic concurrency compares the checkpoint revision before saving. A
  stale worker must stop rather than overwrite newer progress.
- Version workflow definitions. Resuming old state with reordered or changed
  steps can corrupt data, so this lesson fingerprints step names and versions.
- Retry narrow, transient failures only. Invalid input, authentication failure,
  and idempotency conflicts need correction rather than repeated execution.
- Production retries should use exponential backoff, bounded jitter, deadlines,
  and cancellation. Tests should inject delay and randomness instead of sleeping.
- State and effect payloads use canonical JSON here. Real systems also need size
  limits, schema migrations, encryption, retention, and access controls.

Run this file with Python 3.10+.
It uses only the standard library, opens no sockets, sleeps for no time, and
writes no files.

Practice exercises
------------------
1. Implement retry_delays(base, attempts, cap) as a deterministic exponential
   schedule. Reject Booleans, non-positive values, and attempts below one.
2. Implement pending_items(items, completed) while preserving order and
   rejecting duplicate item identifiers.
3. Implement effect_key(workflow_id, step_name, item_id) with normalized,
   length-prefixed components so ambiguous concatenations cannot collide.

Solutions appear below the main example.

Expert challenge: durable import pipeline
-----------------------------------------
Build a package that imports a large file in resumable chunks. Store checkpoints
and idempotency records in SQLite, claim work with expiring leases, quarantine
invalid rows, and publish an outbox event in the same transaction as each
committed chunk. Add a worker CLI with deadlines, bounded retries, structured
logs, metrics, graceful shutdown, and recovery documentation.

Solution guidance
-----------------
1. Give the workflow, definition, input, chunk, and effect stable identifiers.
2. Put the checkpoint update, imported rows, and outbox record in one database
   transaction; enforce unique keys in the schema.
3. Make lease acquisition compare owner and expiry atomically. Fence stale
   owners with a monotonically increasing token.
4. Deliver outbox rows at least once and let the receiver deduplicate by event
   ID. Move exhausted events to a reviewable dead-letter state.
5. Test interruption before and after each boundary, duplicate delivery,
   conflicting key reuse, expired leases, two competing workers, schema
   migration, cancellation, and restart from a real database file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Protocol


_JSON_OBJECT = dict[str, object]
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")


class WorkflowError(Exception):
    """Base class for expected workflow failures."""


class ValidationError(WorkflowError, ValueError):
    """Input or persisted state does not satisfy the workflow contract."""


class ConflictError(WorkflowError):
    """A stable identifier was reused for incompatible data."""


class ConcurrentUpdateError(WorkflowError):
    """A stale worker tried to replace a newer checkpoint."""


class TransientStepError(WorkflowError):
    """A step failed in a way that its policy permits retrying."""


class StepExecutionError(WorkflowError):
    """A step could not complete within its retry policy."""

    def __init__(self, step_name: str, attempts: int) -> None:
        self.step_name = step_name
        self.attempts = attempts
        super().__init__(
            f"step {step_name!r} failed after {attempts} attempt(s)"
        )


def _clean_name(value: str, *, field: str) -> str:
    """Validate a stable machine-readable identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not _NAME_PATTERN.fullmatch(cleaned):
        raise ValidationError(
            f"{field} must contain lowercase words separated by '.', '-', or '_'"
        )
    if len(cleaned) > 80:
        raise ValidationError(f"{field} must contain at most 80 characters")
    return cleaned


def _positive_int(value: int, *, field: str) -> int:
    """Accept a positive integer while rejecting bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _positive_number(value: int | float, *, field: str) -> float:
    """Return a finite positive float while rejecting bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def canonical_object(value: Mapping[str, object], *, field: str) -> str:
    """Encode a string-keyed JSON object deterministically."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValidationError(f"{field} keys must be strings")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field} must contain JSON-compatible values") from error
    if len(encoded.encode("utf-8")) > 100_000:
        raise ValidationError(f"{field} is too large")
    return encoded


def decode_object(encoded: str, *, field: str) -> _JSON_OBJECT:
    """Decode a persisted JSON object and reject other JSON shapes."""
    if not isinstance(encoded, str):
        raise TypeError(f"{field} must be text")
    try:
        value: object = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValidationError(f"{field} is not valid JSON") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValidationError(f"{field} must encode an object")
    return dict(value)


def fingerprint(encoded: str) -> str:
    """Return a stable digest for already-canonical UTF-8 text."""
    if not isinstance(encoded, str):
        raise TypeError("encoded must be text")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EffectLedger(Protocol):
    """Port for a destination that deduplicates logical effects."""

    def apply_once(
        self,
        key: str,
        payload: Mapping[str, object],
    ) -> bool:
        """Apply a new effect, or return False for an identical replay."""


@dataclass(frozen=True, slots=True)
class StepContext:
    """Metadata and effect boundary supplied to one step attempt."""

    workflow_id: str
    step_name: str
    attempt: int
    effects: EffectLedger


StepAction = Callable[
    [StepContext, Mapping[str, object]],
    Mapping[str, object],
]


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """A named, versioned transformation in a workflow definition."""

    name: str
    version: int
    action: StepAction

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field="step name"))
        object.__setattr__(
            self,
            "version",
            _positive_int(self.version, field="step version"),
        )
        if not callable(self.action):
            raise TypeError("step action must be callable")


@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """Serializable progress protected by an optimistic revision."""

    workflow_id: str
    definition_fingerprint: str
    input_fingerprint: str
    next_step: int
    state_json: str
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workflow_id",
            _clean_name(self.workflow_id, field="workflow_id"),
        )
        for field_name, value in (
            ("definition_fingerprint", self.definition_fingerprint),
            ("input_fingerprint", self.input_fingerprint),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValidationError(f"{field_name} must be a SHA-256 hex digest")
        if (
            isinstance(self.next_step, bool)
            or not isinstance(self.next_step, int)
            or self.next_step < 0
        ):
            raise ValidationError("next_step must be a non-negative integer")
        decode_object(self.state_json, field="state_json")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValidationError("revision must be a non-negative integer")

    def state(self) -> _JSON_OBJECT:
        """Return a fresh mutable copy of persisted state."""
        return decode_object(self.state_json, field="state_json")


class CheckpointStore(Protocol):
    """Port for durable compare-and-swap checkpoint storage."""

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Load the latest checkpoint for one workflow."""

    def save(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_revision: int,
    ) -> None:
        """Store checkpoint only when the current revision matches."""


class InMemoryCheckpointStore:
    """Deterministic checkpoint adapter used by examples and tests."""

    def __init__(self) -> None:
        self._values: dict[str, WorkflowCheckpoint] = {}

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        workflow_id = _clean_name(workflow_id, field="workflow_id")
        return self._values.get(workflow_id)

    def save(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_revision: int,
    ) -> None:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < -1
        ):
            raise ValueError("expected_revision must be -1 or greater")
        current = self._values.get(checkpoint.workflow_id)
        current_revision = -1 if current is None else current.revision
        if current_revision != expected_revision:
            raise ConcurrentUpdateError(
                f"expected revision {expected_revision}, found {current_revision}"
            )
        if checkpoint.revision != expected_revision + 1:
            raise ValidationError("new revision must follow expected revision")
        self._values[checkpoint.workflow_id] = checkpoint

    def snapshot(self) -> dict[str, WorkflowCheckpoint]:
        """Return a defensive copy for diagnostics and tests."""
        return dict(self._values)


class InMemoryEffectLedger:
    """Idempotent effect adapter with payload conflict detection."""

    def __init__(self) -> None:
        self._payloads: dict[str, str] = {}
        self._applications: dict[str, int] = {}

    def apply_once(
        self,
        key: str,
        payload: Mapping[str, object],
    ) -> bool:
        key = _clean_name(key, field="effect key")
        encoded = canonical_object(payload, field="effect payload")
        previous = self._payloads.get(key)
        if previous is not None:
            if previous != encoded:
                raise ConflictError(
                    f"effect key {key!r} was reused with different input"
                )
            return False
        self._payloads[key] = encoded
        self._applications[key] = 1
        return True

    def application_count(self, key: str) -> int:
        """Report physical applications for a logical effect key."""
        return self._applications.get(key, 0)


class RunStatus(Enum):
    """Outcome of one bounded runner invocation."""

    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RunReport:
    """Summary returned without exposing mutable checkpoint internals."""

    workflow_id: str
    status: RunStatus
    completed_steps: int
    total_steps: int
    revision: int
    state: Mapping[str, object]


class WorkflowRunner:
    """Execute versioned steps with checkpoints and bounded retries."""

    def __init__(
        self,
        store: CheckpointStore,
        effects: EffectLedger,
        *,
        retry_limit: int = 2,
    ) -> None:
        if not hasattr(store, "load") or not hasattr(store, "save"):
            raise TypeError("store must implement load and save")
        if not hasattr(effects, "apply_once"):
            raise TypeError("effects must implement apply_once")
        if (
            isinstance(retry_limit, bool)
            or not isinstance(retry_limit, int)
            or retry_limit < 0
        ):
            raise ValueError("retry_limit must be a non-negative integer")
        self._store = store
        self._effects = effects
        self._retry_limit = retry_limit

    def run(
        self,
        workflow_id: str,
        steps: Iterable[WorkflowStep],
        initial_state: Mapping[str, object],
        *,
        max_steps: int | None = None,
    ) -> RunReport:
        """Start or resume a workflow and execute at most max_steps."""
        workflow_id = _clean_name(workflow_id, field="workflow_id")
        definition = tuple(steps)
        if not definition:
            raise ValidationError("workflow must contain at least one step")
        names = tuple(step.name for step in definition)
        if len(set(names)) != len(names):
            raise ValidationError("step names must be unique")
        if max_steps is not None:
            max_steps = _positive_int(max_steps, field="max_steps")

        definition_json = json.dumps(
            [(step.name, step.version) for step in definition],
            separators=(",", ":"),
        )
        definition_digest = fingerprint(definition_json)
        initial_json = canonical_object(initial_state, field="initial_state")
        input_digest = fingerprint(initial_json)

        checkpoint = self._store.load(workflow_id)
        if checkpoint is None:
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_id,
                definition_fingerprint=definition_digest,
                input_fingerprint=input_digest,
                next_step=0,
                state_json=initial_json,
                revision=0,
            )
            self._store.save(checkpoint, expected_revision=-1)
        else:
            if checkpoint.definition_fingerprint != definition_digest:
                raise ConflictError("workflow definition changed after it started")
            if checkpoint.input_fingerprint != input_digest:
                raise ConflictError("workflow input changed after it started")
            if checkpoint.next_step > len(definition):
                raise ValidationError("checkpoint points beyond the workflow")

        budget = len(definition) if max_steps is None else max_steps
        executed = 0
        while checkpoint.next_step < len(definition) and executed < budget:
            step = definition[checkpoint.next_step]
            state = checkpoint.state()
            attempts = 0
            while True:
                attempts += 1
                context = StepContext(
                    workflow_id=workflow_id,
                    step_name=step.name,
                    attempt=attempts,
                    effects=self._effects,
                )
                try:
                    updated = step.action(context, state)
                    updated_json = canonical_object(
                        updated,
                        field=f"output from {step.name}",
                    )
                    break
                except TransientStepError as error:
                    if attempts > self._retry_limit:
                        raise StepExecutionError(step.name, attempts) from error

            replacement = WorkflowCheckpoint(
                workflow_id=workflow_id,
                definition_fingerprint=definition_digest,
                input_fingerprint=input_digest,
                next_step=checkpoint.next_step + 1,
                state_json=updated_json,
                revision=checkpoint.revision + 1,
            )
            self._store.save(
                replacement,
                expected_revision=checkpoint.revision,
            )
            checkpoint = replacement
            executed += 1

        status = (
            RunStatus.COMPLETED
            if checkpoint.next_step == len(definition)
            else RunStatus.PAUSED
        )
        return RunReport(
            workflow_id=workflow_id,
            status=status,
            completed_steps=checkpoint.next_step,
            total_steps=len(definition),
            revision=checkpoint.revision,
            state=checkpoint.state(),
        )


def normalize_recipients(
    context: StepContext,
    state: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate, normalize, and deduplicate recipient addresses."""
    del context
    raw = state.get("recipients")
    if not isinstance(raw, list):
        raise ValidationError("recipients must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or "@" not in value:
            raise ValidationError("each recipient must be an email-like string")
        address = value.strip().casefold()
        if not address or len(address) > 254:
            raise ValidationError("recipient is empty or too long")
        if address not in seen:
            seen.add(address)
            normalized.append(address)
    if not normalized:
        raise ValidationError("at least one recipient is required")
    return {"recipients": normalized}


def prepare_summary(
    context: StepContext,
    state: Mapping[str, object],
) -> Mapping[str, object]:
    """Build a deterministic delivery payload from normalized state."""
    del context
    recipients = state.get("recipients")
    if not isinstance(recipients, list):
        raise ValidationError("normalized recipients are missing")
    return {
        "recipients": list(recipients),
        "subject": "Daily workflow summary",
        "recipient_count": len(recipients),
    }


class FlakyDelivery:
    """Simulate a crash after an effect but before its checkpoint."""

    def __init__(self) -> None:
        self.attempts = 0

    def __call__(
        self,
        context: StepContext,
        state: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.attempts += 1
        payload = {
            "recipients": state.get("recipients"),
            "subject": state.get("subject"),
        }
        context.effects.apply_once(
            f"{context.workflow_id}.{context.step_name}",
            payload,
        )
        if self.attempts == 1:
            raise TransientStepError("simulated interruption after delivery")
        return {**state, "delivered": True}


# Practice exercise solutions


def retry_delays(
    base: int | float,
    attempts: int,
    cap: int | float,
) -> tuple[float, ...]:
    """Solution 1: return a capped deterministic exponential schedule."""
    base_value = _positive_number(base, field="base")
    cap_value = _positive_number(cap, field="cap")
    attempts = _positive_int(attempts, field="attempts")
    return tuple(min(cap_value, base_value * (2**index)) for index in range(attempts))


def pending_items(
    items: Iterable[str],
    completed: Iterable[str],
) -> tuple[str, ...]:
    """Solution 2: preserve order and reject ambiguous duplicate IDs."""
    completed_set = {_clean_name(item, field="completed item") for item in completed}
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = _clean_name(item, field="item")
        if item in seen:
            raise ValidationError(f"duplicate item identifier: {item}")
        seen.add(item)
        if item not in completed_set:
            result.append(item)
    return tuple(result)


def effect_key(workflow_id: str, step_name: str, item_id: str) -> str:
    """Solution 3: build an unambiguous normalized effect identifier."""
    components = (
        _clean_name(workflow_id, field="workflow_id"),
        _clean_name(step_name, field="step_name"),
        _clean_name(item_id, field="item_id"),
    )
    raw = "|".join(f"{len(value)}:{value}" for value in components)
    return "effect." + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _workflow(delivery: FlakyDelivery) -> tuple[WorkflowStep, ...]:
    """Create the example workflow with explicit definition versions."""
    return (
        WorkflowStep("normalize", 1, normalize_recipients),
        WorkflowStep("prepare", 1, prepare_summary),
        WorkflowStep("deliver", 1, delivery),
    )


def run_self_checks() -> None:
    """Exercise success, replay, pause, conflict, and stale-worker paths."""
    store = InMemoryCheckpointStore()
    effects = InMemoryEffectLedger()
    delivery = FlakyDelivery()
    runner = WorkflowRunner(store, effects, retry_limit=1)
    initial = {
        "recipients": [
            " Ada@Example.com ",
            "bob@example.com",
            "ada@example.com",
        ]
    }

    report = runner.run("daily.summary", _workflow(delivery), initial)
    assert report.status is RunStatus.COMPLETED
    assert report.completed_steps == 3
    assert report.state["recipient_count"] == 2
    assert report.state["delivered"] is True
    assert delivery.attempts == 2
    assert effects.application_count("daily.summary.deliver") == 1

    replay = runner.run("daily.summary", _workflow(delivery), initial)
    assert replay.status is RunStatus.COMPLETED
    assert replay.revision == report.revision
    assert delivery.attempts == 2
    assert effects.application_count("daily.summary.deliver") == 1

    paused_store = InMemoryCheckpointStore()
    paused_effects = InMemoryEffectLedger()
    paused_delivery = FlakyDelivery()
    paused_runner = WorkflowRunner(paused_store, paused_effects, retry_limit=1)
    paused = paused_runner.run(
        "weekly.summary",
        _workflow(paused_delivery),
        initial,
        max_steps=1,
    )
    assert paused.status is RunStatus.PAUSED
    assert paused.completed_steps == 1
    resumed = paused_runner.run(
        "weekly.summary",
        _workflow(paused_delivery),
        initial,
    )
    assert resumed.status is RunStatus.COMPLETED
    assert paused_effects.application_count("weekly.summary.deliver") == 1

    for changed_input, changed_steps in (
        ({"recipients": ["different@example.com"]}, _workflow(delivery)),
        (
            initial,
            (
                WorkflowStep("normalize", 2, normalize_recipients),
                WorkflowStep("prepare", 1, prepare_summary),
                WorkflowStep("deliver", 1, delivery),
            ),
        ),
    ):
        try:
            runner.run("daily.summary", changed_steps, changed_input)
        except ConflictError:
            pass
        else:
            raise AssertionError("changed input or definition must conflict")

    checkpoint = store.load("daily.summary")
    assert checkpoint is not None
    try:
        store.save(checkpoint, expected_revision=checkpoint.revision - 1)
    except ConcurrentUpdateError:
        pass
    else:
        raise AssertionError("stale checkpoint updates must fail")

    conflict_ledger = InMemoryEffectLedger()
    assert conflict_ledger.apply_once("order.charge", {"amount": 10})
    assert not conflict_ledger.apply_once("order.charge", {"amount": 10})
    try:
        conflict_ledger.apply_once("order.charge", {"amount": 11})
    except ConflictError:
        pass
    else:
        raise AssertionError("changed payload under one key must conflict")

    assert retry_delays(0.5, 5, 3) == (0.5, 1.0, 2.0, 3.0, 3.0)
    assert pending_items(
        ["chunk.1", "chunk.2", "chunk.3"],
        ["chunk.2"],
    ) == ("chunk.1", "chunk.3")
    assert effect_key("daily.summary", "deliver", "batch.7") == effect_key(
        "daily.summary",
        "deliver",
        "batch.7",
    )
    assert effect_key("a.b", "c", "d") != effect_key("a", "b.c", "d")

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: WorkflowStep("Bad Name", 1, normalize_recipients),
        lambda: WorkflowStep("valid", True, normalize_recipients),
        lambda: runner.run("empty.workflow", (), {}),
        lambda: runner.run(
            "duplicate.steps",
            (
                WorkflowStep("same", 1, normalize_recipients),
                WorkflowStep("same", 1, normalize_recipients),
            ),
            {},
        ),
        lambda: runner.run(
            "zero.budget",
            (WorkflowStep("valid", 1, normalize_recipients),),
            {},
            max_steps=0,
        ),
        lambda: retry_delays(True, 2, 5),
        lambda: pending_items(["chunk.1", "chunk.1"], []),
        lambda: canonical_object({"bad": float("nan")}, field="bad"),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError, WorkflowError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a safe demonstration of interruption and idempotent recovery."""
    run_self_checks()

    store = InMemoryCheckpointStore()
    effects = InMemoryEffectLedger()
    delivery = FlakyDelivery()
    report = WorkflowRunner(store, effects, retry_limit=1).run(
        "demo.summary",
        _workflow(delivery),
        {"recipients": ["learner@example.com", "mentor@example.com"]},
    )

    print("Status:", report.status.value)
    print("Completed steps:", f"{report.completed_steps}/{report.total_steps}")
    print("Delivery attempts:", delivery.attempts)
    print("Physical deliveries:", effects.application_count("demo.summary.deliver"))
    print("Retry schedule:", retry_delays(0.5, 5, 4))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
