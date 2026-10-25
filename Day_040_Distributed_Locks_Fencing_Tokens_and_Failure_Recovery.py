"""Day 40: Distributed locks, fencing tokens, and failure recovery.

Learning goals
--------------
1. Explain why a lease alone cannot stop a paused process from writing late.
2. Issue monotonically increasing fencing tokens for every new lock ownership.
3. Reject stale owners at the protected resource, not only at the lock service.
4. Make acquisition and side effects safe to retry after ambiguous failures.
5. Recover work after a crash without silently applying an operation twice.

Teaching notes
--------------
- A distributed lock coordinates contenders, but it cannot revoke code that is
  paused by a long garbage collection, machine suspension, or network partition.
  That code may resume after its lease expires and still attempt a write.
- A fencing token is a monotonically increasing ownership number. Every protected
  write carries the token, and the resource rejects tokens older than the newest
  one it has observed. The check belongs in the database, object store, or other
  side-effect boundary.
- Lease expiry creates availability; fencing creates safety. Neither replaces
  idempotency. A retry after a timeout may repeat an operation whose first result
  was lost, so stable operation ids must return the original result.
- Lock acquisition needs an idempotency key too. If a client times out after the
  lock service grants ownership, retrying the same acquisition request should
  return the same grant instead of consuming another fencing token.
- Use a monotonic clock for lease duration. Wall-clock time is appropriate for
  human schedules and audit timestamps, but clock corrections must not extend or
  shorten ownership unexpectedly.
- Choose lock scope carefully. A global lock limits throughput; a resource-level
  lock permits safe parallelism. Acquire multiple locks in one documented order,
  or avoid multi-lock transactions, to prevent deadlocks.
- Never infer ownership from a process id or hostname. Present the complete grant
  on renew and release, and compare the resource, owner, acquisition id, fencing
  token, and unexpired deadline atomically.
- Recovery is a protocol, not an exception handler. Persist intent, use an
  idempotent side effect, record completion, and make every transition observable.
  Define what operators do when the lock service or protected resource is down.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, writes no files,
and advances a manual clock instead of sleeping.

Practice exercises
------------------
1. Implement lease_remaining(grant, now). Reject invalid time values and clamp an
   expired lease to 0.0.
2. Implement is_newer_fence(candidate, last_seen). Accept positive integer
   fencing tokens, reject Booleans, and report whether candidate is newer.
3. Implement recovery_needed(completed, side_effect_visible). Return True only
   when a visible side effect has not been recorded as complete.

Solutions appear in the executable code below.

Expert challenge: recoverable distributed maintenance service
--------------------------------------------------------------
Build a multi-process maintenance service backed by PostgreSQL or another store
that supports atomic compare-and-set updates. Persist operation intent, acquire a
resource-scoped lease with an idempotent acquisition id, and require fencing
tokens on every mutation. Add bounded lease renewal, cancellation, retry budgets,
dead-letter inspection, audit events, and metrics for contention, expired grants,
stale-write rejections, recovery age, and replayed operations. Test process pause,
network partition, lost responses, lock-service restart, two recoverers racing,
and a crash after the side effect but before completion is recorded. Document the
consistency model, token durability, lock ordering, and operator runbook.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import RLock
from typing import Protocol


class CoordinationError(RuntimeError):
    """Base class for lock and recovery protocol failures."""


class ValidationError(CoordinationError, ValueError):
    """Raised when public input violates a protocol invariant."""


class LockBusy(CoordinationError):
    """Raised when another live owner holds the requested resource."""


class OwnershipLost(CoordinationError):
    """Raised when a grant is expired, released, or no longer current."""


class StaleFence(CoordinationError):
    """Raised when a protected write presents an obsolete fencing token."""


class OperationConflict(CoordinationError):
    """Raised when an idempotency key is reused for different work."""


class SimulatedCrash(CoordinationError):
    """Raised by the demonstration at a deliberate crash boundary."""


class Clock(Protocol):
    """A monotonic source of elapsed time."""

    def now(self) -> float:
        """Return a finite monotonic timestamp."""


def _clean_text(value: object, *, field: str, limit: int = 120) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValidationError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise ValidationError(f"{field} must be at most {limit} characters")
    return cleaned


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def _positive_float(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0:
        raise ValidationError(f"{field} must be positive")
    return number


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class LockGrant:
    """Proof of one time-bounded ownership decision."""

    resource: str
    owner: str
    acquisition_id: str
    fencing_token: int
    lease_until: float


@dataclass(frozen=True, slots=True)
class LockView:
    """A safe diagnostic view of current lock state."""

    resource: str
    owner: str | None
    acquisition_id: str | None
    fencing_token: int
    lease_until: float | None
    active: bool


@dataclass(slots=True)
class _LockState:
    fencing_token: int = 0
    owner: str | None = None
    acquisition_id: str | None = None
    lease_until: float | None = None


class ManualClock:
    """A deterministic monotonic clock for examples and tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._time = _finite(start, field="start")
        self._lock = RLock()

    def now(self) -> float:
        with self._lock:
            return self._time

    def advance(self, seconds: float) -> float:
        delta = _finite(seconds, field="seconds")
        if delta < 0:
            raise ValidationError("seconds must not be negative")
        with self._lock:
            self._time += delta
            return self._time


class FencedLockService:
    """In-memory lease service that issues durable-style fencing numbers."""

    def __init__(self, clock: Clock) -> None:
        if not callable(getattr(clock, "now", None)):
            raise ValidationError("clock must provide now()")
        self._clock = clock
        self._states: dict[str, _LockState] = {}
        self._lock = RLock()

    def acquire(
        self,
        resource: str,
        owner: str,
        acquisition_id: str,
        *,
        ttl: float,
    ) -> LockGrant:
        """Acquire a free or expired resource, or replay the same request."""
        resource_name = _clean_text(resource, field="resource")
        owner_name = _clean_text(owner, field="owner")
        request_id = _clean_text(acquisition_id, field="acquisition_id")
        duration = _positive_float(ttl, field="ttl")
        now = _finite(self._clock.now(), field="clock.now()")

        with self._lock:
            state = self._states.setdefault(resource_name, _LockState())
            active = state.lease_until is not None and state.lease_until > now

            if active:
                if (
                    state.owner == owner_name
                    and state.acquisition_id == request_id
                ):
                    return self._grant(resource_name, state)
                raise LockBusy(
                    f"{resource_name!r} is held by another live acquisition"
                )

            state.fencing_token += 1
            state.owner = owner_name
            state.acquisition_id = request_id
            state.lease_until = now + duration
            return self._grant(resource_name, state)

    def renew(self, grant: LockGrant, *, ttl: float) -> LockGrant:
        """Extend a current grant without issuing a new fencing token."""
        self._validate_grant_type(grant)
        duration = _positive_float(ttl, field="ttl")
        now = _finite(self._clock.now(), field="clock.now()")

        with self._lock:
            state = self._require_current(grant, now=now)
            state.lease_until = now + duration
            return self._grant(grant.resource, state)

    def release(self, grant: LockGrant) -> None:
        """Release only the exact current grant; stale owners cannot unlock."""
        self._validate_grant_type(grant)
        now = _finite(self._clock.now(), field="clock.now()")

        with self._lock:
            state = self._require_current(grant, now=now)
            state.owner = None
            state.acquisition_id = None
            state.lease_until = None

    def inspect(self, resource: str) -> LockView:
        """Return lock state without exposing mutable service internals."""
        resource_name = _clean_text(resource, field="resource")
        now = _finite(self._clock.now(), field="clock.now()")
        with self._lock:
            state = self._states.get(resource_name, _LockState())
            active = state.lease_until is not None and state.lease_until > now
            return LockView(
                resource=resource_name,
                owner=state.owner if active else None,
                acquisition_id=state.acquisition_id if active else None,
                fencing_token=state.fencing_token,
                lease_until=state.lease_until if active else None,
                active=active,
            )

    @staticmethod
    def _validate_grant_type(grant: LockGrant) -> None:
        if not isinstance(grant, LockGrant):
            raise ValidationError("grant must be LockGrant")

    @staticmethod
    def _grant(resource: str, state: _LockState) -> LockGrant:
        if (
            state.owner is None
            or state.acquisition_id is None
            or state.lease_until is None
        ):
            raise AssertionError("active lock state is incomplete")
        return LockGrant(
            resource=resource,
            owner=state.owner,
            acquisition_id=state.acquisition_id,
            fencing_token=state.fencing_token,
            lease_until=state.lease_until,
        )

    def _require_current(self, grant: LockGrant, *, now: float) -> _LockState:
        state = self._states.get(grant.resource)
        if (
            state is None
            or state.owner != grant.owner
            or state.acquisition_id != grant.acquisition_id
            or state.fencing_token != grant.fencing_token
            or state.lease_until != grant.lease_until
            or state.lease_until is None
            or state.lease_until <= now
        ):
            raise OwnershipLost("the lock grant is no longer current")
        return state


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """One committed view of a fenced resource."""

    resource: str
    value: str
    version: int
    last_fencing_token: int
    operation_id: str


@dataclass(slots=True)
class _ResourceState:
    value: str = ""
    version: int = 0
    last_fencing_token: int = 0
    operation_id: str = ""


@dataclass(frozen=True, slots=True)
class _AppliedOperation:
    resource: str
    value: str
    version: int


class FencedResourceStore:
    """Side-effect boundary that enforces fences and idempotent operations."""

    def __init__(self) -> None:
        self._resources: dict[str, _ResourceState] = {}
        self._operations: dict[str, _AppliedOperation] = {}
        self._lock = RLock()

    def apply(
        self,
        resource: str,
        value: str,
        *,
        fencing_token: int,
        operation_id: str,
    ) -> ResourceSnapshot:
        """Apply once, replay safely, and reject any older ownership epoch."""
        resource_name = _clean_text(resource, field="resource")
        new_value = _clean_text(value, field="value", limit=500)
        token = _positive_int(fencing_token, field="fencing_token")
        op_id = _clean_text(operation_id, field="operation_id")

        with self._lock:
            state = self._resources.setdefault(resource_name, _ResourceState())
            if token < state.last_fencing_token:
                raise StaleFence(
                    f"token {token} is older than {state.last_fencing_token}"
                )

            previous = self._operations.get(op_id)
            if previous is not None:
                if (
                    previous.resource != resource_name
                    or previous.value != new_value
                ):
                    raise OperationConflict(
                        "operation_id was already used for different work"
                    )
                state.last_fencing_token = max(
                    state.last_fencing_token,
                    token,
                )
                return self._snapshot(resource_name, state)

            state.value = new_value
            state.version += 1
            state.last_fencing_token = token
            state.operation_id = op_id
            self._operations[op_id] = _AppliedOperation(
                resource=resource_name,
                value=new_value,
                version=state.version,
            )
            return self._snapshot(resource_name, state)

    def has_operation(self, operation_id: str) -> bool:
        """Return whether an idempotent side effect is already visible."""
        op_id = _clean_text(operation_id, field="operation_id")
        with self._lock:
            return op_id in self._operations

    def get(self, resource: str) -> ResourceSnapshot:
        resource_name = _clean_text(resource, field="resource")
        with self._lock:
            state = self._resources.get(resource_name)
            if state is None or state.version == 0:
                raise KeyError(resource_name)
            return self._snapshot(resource_name, state)

    @staticmethod
    def _snapshot(resource: str, state: _ResourceState) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource=resource,
            value=state.value,
            version=state.version,
            last_fencing_token=state.last_fencing_token,
            operation_id=state.operation_id,
        )


@dataclass(frozen=True, slots=True)
class Operation:
    """Stable intent for one recoverable resource update."""

    operation_id: str
    resource: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _clean_text(self.operation_id, field="operation_id"),
        )
        object.__setattr__(
            self,
            "resource",
            _clean_text(self.resource, field="resource"),
        )
        object.__setattr__(
            self,
            "value",
            _clean_text(self.value, field="value", limit=500),
        )


class CompletionLedger:
    """Thread-safe record of operations whose completion was acknowledged."""

    def __init__(self) -> None:
        self._completed: set[str] = set()
        self._lock = RLock()

    def is_complete(self, operation_id: str) -> bool:
        op_id = _clean_text(operation_id, field="operation_id")
        with self._lock:
            return op_id in self._completed

    def mark_complete(self, operation_id: str) -> None:
        op_id = _clean_text(operation_id, field="operation_id")
        with self._lock:
            self._completed.add(op_id)


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Outcome of a normal run or an idempotent recovery."""

    snapshot: ResourceSnapshot
    recovered: bool


class RecoveryRunner:
    """Execute intent through a lock, fenced store, and completion ledger."""

    def __init__(
        self,
        locks: FencedLockService,
        resources: FencedResourceStore,
        ledger: CompletionLedger,
    ) -> None:
        self._locks = locks
        self._resources = resources
        self._ledger = ledger

    def run(
        self,
        operation: Operation,
        *,
        owner: str,
        acquisition_id: str,
        ttl: float = 10.0,
        crash_after_apply: bool = False,
    ) -> RecoveryResult:
        """Apply work once and expose the crash window for deterministic tests."""
        if not isinstance(operation, Operation):
            raise ValidationError("operation must be Operation")
        if not isinstance(crash_after_apply, bool):
            raise ValidationError("crash_after_apply must be Boolean")

        if self._ledger.is_complete(operation.operation_id):
            return RecoveryResult(
                snapshot=self._resources.get(operation.resource),
                recovered=False,
            )

        grant = self._locks.acquire(
            operation.resource,
            owner,
            acquisition_id,
            ttl=ttl,
        )
        replaying_visible_effect = self._resources.has_operation(
            operation.operation_id
        )
        snapshot = self._resources.apply(
            operation.resource,
            operation.value,
            fencing_token=grant.fencing_token,
            operation_id=operation.operation_id,
        )

        if crash_after_apply:
            raise SimulatedCrash(
                "process stopped after side effect and before completion record"
            )

        self._ledger.mark_complete(operation.operation_id)
        self._locks.release(grant)
        return RecoveryResult(
            snapshot=snapshot,
            recovered=replaying_visible_effect,
        )


def lease_remaining(grant: LockGrant, now: float) -> float:
    """Solution 1: return the non-negative lifetime left on a grant."""
    if not isinstance(grant, LockGrant):
        raise ValidationError("grant must be LockGrant")
    current = _finite(now, field="now")
    return max(0.0, grant.lease_until - current)


def is_newer_fence(candidate: int, last_seen: int) -> bool:
    """Solution 2: compare validated positive fencing tokens."""
    new_token = _positive_int(candidate, field="candidate")
    previous = _positive_int(last_seen, field="last_seen")
    return new_token > previous


def recovery_needed(completed: bool, side_effect_visible: bool) -> bool:
    """Solution 3: identify the ambiguous post-side-effect crash state."""
    if not isinstance(completed, bool):
        raise ValidationError("completed must be Boolean")
    if not isinstance(side_effect_visible, bool):
        raise ValidationError("side_effect_visible must be Boolean")
    return side_effect_visible and not completed


def run_self_checks() -> None:
    """Verify contention, fencing, renewal, idempotency, and crash recovery."""
    clock = ManualClock(100.0)
    locks = FencedLockService(clock)
    resources = FencedResourceStore()

    first = locks.acquire(
        "customer:42",
        "worker-a",
        "acquire-a",
        ttl=5.0,
    )
    replay = locks.acquire(
        "customer:42",
        "worker-a",
        "acquire-a",
        ttl=5.0,
    )
    assert replay == first
    assert first.fencing_token == 1
    assert lease_remaining(first, clock.now()) == 5.0
    assert locks.inspect("customer:42").active is True

    try:
        locks.acquire(
            "customer:42",
            "worker-b",
            "acquire-b",
            ttl=5.0,
        )
    except LockBusy:
        pass
    else:
        raise AssertionError("a live owner must exclude another acquisition")

    initial_write = resources.apply(
        "customer:42",
        "plan=standard",
        fencing_token=first.fencing_token,
        operation_id="change-plan-001",
    )
    assert initial_write.version == 1

    renewed = locks.renew(first, ttl=8.0)
    assert renewed.fencing_token == first.fencing_token
    assert renewed.lease_until == 108.0
    assert lease_remaining(renewed, 109.0) == 0.0

    clock.advance(8.0)
    second = locks.acquire(
        "customer:42",
        "worker-b",
        "acquire-b",
        ttl=5.0,
    )
    assert second.fencing_token == 2
    assert is_newer_fence(second.fencing_token, first.fencing_token)

    replayed_write = resources.apply(
        "customer:42",
        "plan=standard",
        fencing_token=second.fencing_token,
        operation_id="change-plan-001",
    )
    assert replayed_write.version == 1
    assert replayed_write.last_fencing_token == 2

    try:
        resources.apply(
            "customer:42",
            "plan=premium",
            fencing_token=first.fencing_token,
            operation_id="stale-change",
        )
    except StaleFence:
        pass
    else:
        raise AssertionError("an old owner must be fenced at the resource")

    try:
        locks.release(first)
    except OwnershipLost:
        pass
    else:
        raise AssertionError("a stale grant must not release a newer lock")

    current_write = resources.apply(
        "customer:42",
        "plan=premium",
        fencing_token=second.fencing_token,
        operation_id="change-plan-002",
    )
    assert current_write.version == 2
    locks.release(second)
    assert locks.inspect("customer:42").active is False

    ledger = CompletionLedger()
    runner = RecoveryRunner(locks, resources, ledger)
    operation = Operation(
        "rebuild-index-001",
        "search-index:products",
        "generation=7",
    )

    try:
        runner.run(
            operation,
            owner="worker-crash",
            acquisition_id="attempt-1",
            ttl=4.0,
            crash_after_apply=True,
        )
    except SimulatedCrash:
        pass
    else:
        raise AssertionError("the crash boundary must be observable")

    assert ledger.is_complete(operation.operation_id) is False
    assert recovery_needed(False, True) is True
    assert resources.get(operation.resource).version == 1

    clock.advance(4.0)
    recovered = runner.run(
        operation,
        owner="worker-recovery",
        acquisition_id="attempt-2",
        ttl=4.0,
    )
    assert recovered.snapshot.version == 1
    assert recovered.snapshot.last_fencing_token == 2
    assert recovered.recovered is True
    assert ledger.is_complete(operation.operation_id) is True

    repeated = runner.run(
        operation,
        owner="worker-late-retry",
        acquisition_id="attempt-3",
    )
    assert repeated.snapshot.version == 1
    assert repeated.recovered is False

    conflict_grant = locks.acquire(
        "conflict-resource",
        "worker-c",
        "conflict-acquire",
        ttl=2.0,
    )
    resources.apply(
        "conflict-resource",
        "value=one",
        fencing_token=conflict_grant.fencing_token,
        operation_id="shared-operation-id",
    )
    try:
        resources.apply(
            "conflict-resource",
            "value=two",
            fencing_token=conflict_grant.fencing_token,
            operation_id="shared-operation-id",
        )
    except OperationConflict:
        pass
    else:
        raise AssertionError("idempotency keys must not alias different work")

    invalid_calls = (
        lambda: ManualClock(float("nan")),
        lambda: clock.advance(-1),
        lambda: locks.acquire("", "owner", "request", ttl=1),
        lambda: locks.acquire("resource", "", "request", ttl=1),
        lambda: locks.acquire("resource", "owner", "", ttl=1),
        lambda: locks.acquire("resource", "owner", "request", ttl=True),
        lambda: lease_remaining(first, float("inf")),
        lambda: is_newer_fence(True, 1),
        lambda: is_newer_fence(2, 0),
        lambda: recovery_needed(1, True),
        lambda: Operation("", "resource", "value"),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (ValidationError, CoordinationError):
            pass
        else:
            raise AssertionError("invalid input must raise a protocol error")


def main() -> None:
    """Run a deterministic stale-owner and recovery demonstration."""
    run_self_checks()

    clock = ManualClock()
    locks = FencedLockService(clock)
    resources = FencedResourceStore()

    old_grant = locks.acquire(
        "report:monthly",
        "worker-a",
        "request-a",
        ttl=3.0,
    )
    resources.apply(
        old_grant.resource,
        "status=building",
        fencing_token=old_grant.fencing_token,
        operation_id="report-start",
    )

    clock.advance(3.0)
    new_grant = locks.acquire(
        "report:monthly",
        "worker-b",
        "request-b",
        ttl=3.0,
    )
    final = resources.apply(
        new_grant.resource,
        "status=complete",
        fencing_token=new_grant.fencing_token,
        operation_id="report-finish",
    )

    try:
        resources.apply(
            old_grant.resource,
            "status=stale-overwrite",
            fencing_token=old_grant.fencing_token,
            operation_id="late-write",
        )
    except StaleFence as error:
        print("Rejected stale owner:", error)

    print("Current resource:", final)
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
