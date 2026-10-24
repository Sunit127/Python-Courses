"""Day 39: Durable scheduling, leases, and distributed coordination.

Learning goals
--------------
1. Persist a scheduled job before acknowledging it to a caller.
2. Claim due work with an expiring lease so another worker can recover it.
3. Use compare-and-set style ownership checks for renew, complete, and retry.
4. Separate monotonic lease time from wall-clock scheduling decisions.
5. Design retries, fencing, and recovery around crashes and duplicate messages.

Teaching notes
--------------
- A task queue keeps work moving while a process is alive. A durable scheduler
  also records the job and its next run time before returning success, so a
  restart can discover work that was accepted.
- A lease is a time-bounded claim, not a permanent lock. If a worker crashes or
  stops renewing, another worker may reclaim the job after the visibility window.
  Every mutation must check the current lease token; a stale worker must be
  rejected even if it wakes up late.
- Use a monotonic clock for elapsed lease deadlines. Wall clocks can jump because
  of NTP corrections or daylight-saving changes. Store a wall-clock due time for
  user-facing schedules, then convert it to a monotonic deadline at the worker
  boundary.
- Claim, renew, complete, and fail should be atomic transactions in a real
  database. This lesson's lock protects an in-memory store so the coordination
  protocol stays visible and deterministic; it is not durable across process
  restarts.
- A retry changes the schedule, not the identity. Keep a stable job id, count
  attempts, use bounded exponential backoff with jitter, and send poison jobs to
  a dead-letter workflow after a policy limit.
- Leases do not prevent a paused worker from writing after its lease expires.
  Production systems use fencing tokens (a monotonically increasing claim
  number) at the side-effect boundary so an old owner cannot overwrite a newer
  owner.
- Idempotency keys and unique constraints make duplicate enqueue and duplicate
  completion safe. Observability should include due lag, lease age, attempts,
  expirations, retry delay, and the reason a claim was rejected.
- A scheduler is a coordination service: define failure semantics first. Decide
  what happens during a database outage, clock skew, deploy, cancellation, and
  partial side effect before choosing a polling interval.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, writes no files,
and advances a manual clock instead of sleeping.

Practice exercises
------------------
1. Implement next_backoff(attempt, base=1.0, cap=60.0). Return bounded
   exponential backoff, reject Boolean or non-positive inputs, and never return
   more than cap.
2. Implement is_due(run_at, now). Validate finite numbers and return whether the
   scheduled time has arrived.
3. Extend LeaseStore with a cancel(job_id) operation that is safe to retry and
   refuses to cancel a completed job.

Solutions appear in the executable code below.

Expert challenge: production scheduler
--------------------------------------
Build a multi-tenant scheduler backed by SQLite or PostgreSQL. Persist a
wall-clock schedule and a monotonic lease deadline, claim rows with a single
atomic UPDATE/RETURNING statement, and issue fencing tokens for every claim.
Add cron-like recurrence, per-tenant quotas, priority aging, bounded retries with
jitter, dead-letter inspection, idempotency keys, clock-skew alarms, and graceful
draining during deploys. Test process crashes between claim and side effect,
lease expiry during a long task, duplicate completion, database failover,
poison jobs, and two schedulers racing for one row. Document the consistency
guarantees and the operational runbook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from threading import Lock
from typing import Protocol


class SchedulerError(RuntimeError):
    """Base class for scheduler protocol failures."""


class ValidationError(SchedulerError, ValueError):
    """Raised when a public operation receives invalid data."""


class JobAlreadyExists(SchedulerError):
    """Raised when an idempotency key is reused for a different enqueue."""


class JobNotFound(SchedulerError):
    """Raised when a requested job does not exist."""


class LeaseConflict(SchedulerError):
    """Raised when a worker presents a stale or incorrect lease token."""


class JobStateError(SchedulerError):
    """Raised when an operation is incompatible with the job state."""


class JobStatus(Enum):
    READY = "ready"
    LEASED = "leased"
    COMPLETE = "complete"
    FAILED = "failed"


class Clock(Protocol):
    """A source of elapsed time used for lease deadlines."""

    def now(self) -> float:
        """Return a finite, monotonically increasing timestamp."""


def _clean_text(value: object, *, field: str, limit: int = 100) -> str:
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


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value < 0:
        raise ValidationError(f"{field} must not be negative")
    return value


@dataclass(frozen=True, slots=True)
class Lease:
    """The token and deadline a worker must present for every mutation."""

    job_id: str
    payload: Mapping[str, object]
    attempt: int
    fencing_token: int
    lease_token: str
    lease_until: float

    def is_active(self, now: float) -> bool:
        """Return whether this lease is live at a caller-supplied timestamp."""
        return self.lease_until > _finite(now, field="now")


@dataclass(frozen=True, slots=True)
class JobView:
    """A safe read model that never exposes mutable store state."""

    job_id: str
    payload: Mapping[str, object]
    run_at: float
    status: JobStatus
    attempts: int
    fencing_token: int
    lease_owner: str | None
    lease_until: float | None


@dataclass(slots=True)
class _Job:
    job_id: str
    payload: Mapping[str, object]
    run_at: float
    status: JobStatus = JobStatus.READY
    attempts: int = 0
    fencing_token: int = 0
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_until: float | None = None


class ManualClock:
    """Deterministic clock for examples and tests; production uses a monotonic clock."""

    def __init__(self, start: float = 0.0) -> None:
        self._time = _finite(start, field="start")

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> float:
        delta = _finite(seconds, field="seconds")
        if delta < 0:
            raise ValidationError("seconds must not be negative")
        self._time += delta
        return self._time


class LeaseStore:
    """Thread-safe coordination protocol; replace the lock with DB transactions in production."""

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = Lock()
        self._token_counter = 0

    def enqueue(
        self,
        job_id: str,
        payload: Mapping[str, object],
        *,
        run_at: float = 0.0,
    ) -> JobView:
        """Durably accept one idempotent job in this store."""
        clean_id = _clean_text(job_id, field="job_id")
        if not isinstance(payload, Mapping):
            raise ValidationError("payload must be a mapping")
        clean_run_at = _finite(run_at, field="run_at")
        copied = dict(payload)
        if any(not isinstance(key, str) for key in copied):
            raise ValidationError("payload keys must be text")
        with self._lock:
            if clean_id in self._jobs:
                raise JobAlreadyExists(f"job {clean_id!r} already exists")
            self._jobs[clean_id] = _Job(
                job_id=clean_id,
                payload=copied,
                run_at=clean_run_at,
            )
            return self._view(self._jobs[clean_id])

    def _view(self, job: _Job) -> JobView:
        return JobView(
            job_id=job.job_id,
            payload=dict(job.payload),
            run_at=job.run_at,
            status=job.status,
            attempts=job.attempts,
            fencing_token=job.fencing_token,
            lease_owner=job.lease_owner,
            lease_until=job.lease_until,
        )

    def _expire_leases(self, now: float) -> None:
        for job in self._jobs.values():
            if (
                job.status is JobStatus.LEASED
                and job.lease_until is not None
                and job.lease_until <= now
            ):
                job.status = JobStatus.READY
                job.lease_owner = None
                job.lease_token = None
                job.lease_until = None

    def claim_due(
        self,
        *,
        worker_id: str,
        now: float,
        lease_seconds: float,
        limit: int = 1,
    ) -> tuple[Lease, ...]:
        """Atomically reclaim expired work and claim the earliest due jobs."""
        clean_worker = _clean_text(worker_id, field="worker_id")
        clean_now = _finite(now, field="now")
        duration = _positive_float(lease_seconds, field="lease_seconds")
        count = _positive_int(limit, field="limit")
        with self._lock:
            self._expire_leases(clean_now)
            candidates = sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.status is JobStatus.READY and job.run_at <= clean_now
                ),
                key=lambda job: (job.run_at, job.job_id),
            )[:count]
            leases: list[Lease] = []
            for job in candidates:
                self._token_counter += 1
                job.status = JobStatus.LEASED
                job.attempts += 1
                job.fencing_token = self._token_counter
                job.lease_owner = clean_worker
                job.lease_token = f"{clean_worker}:{job.fencing_token}"
                job.lease_until = clean_now + duration
                leases.append(self._lease(job))
            return tuple(leases)

    def _lease(self, job: _Job) -> Lease:
        if job.lease_token is None or job.lease_until is None:
            raise JobStateError("job does not have an active lease")
        return Lease(
            job_id=job.job_id,
            payload=dict(job.payload),
            attempt=job.attempts,
            fencing_token=job.fencing_token,
            lease_token=job.lease_token,
            lease_until=job.lease_until,
        )

    def _require_current_lease(
        self,
        job_id: str,
        lease_token: str,
        now: float,
    ) -> _Job:
        clean_id = _clean_text(job_id, field="job_id")
        clean_token = _clean_text(lease_token, field="lease_token")
        clean_now = _finite(now, field="now")
        self._expire_leases(clean_now)
        job = self._jobs.get(clean_id)
        if job is None:
            raise JobNotFound(f"job {clean_id!r} does not exist")
        if (
            job.status is not JobStatus.LEASED
            or job.lease_token != clean_token
            or job.lease_until is None
            or job.lease_until <= clean_now
        ):
            raise LeaseConflict(f"lease for {clean_id!r} is no longer current")
        return job

    def renew(
        self,
        job_id: str,
        lease_token: str,
        *,
        now: float,
        lease_seconds: float,
    ) -> Lease:
        """Extend a lease only when its token and deadline are still current."""
        clean_now = _finite(now, field="now")
        duration = _positive_float(lease_seconds, field="lease_seconds")
        with self._lock:
            job = self._require_current_lease(job_id, lease_token, clean_now)
            job.lease_until = clean_now + duration
            return self._lease(job)

    def complete(self, job_id: str, lease_token: str, *, now: float) -> JobView:
        """Commit success with a compare-and-set lease check."""
        clean_now = _finite(now, field="now")
        with self._lock:
            job = self._require_current_lease(job_id, lease_token, clean_now)
            job.status = JobStatus.COMPLETE
            job.lease_owner = None
            job.lease_token = None
            job.lease_until = None
            return self._view(job)

    def fail(
        self,
        job_id: str,
        lease_token: str,
        *,
        now: float,
        retry_at: float | None = None,
    ) -> JobView:
        """Release the lease for a retry or move the job to a terminal failure."""
        clean_now = _finite(now, field="now")
        clean_retry = None if retry_at is None else _finite(retry_at, field="retry_at")
        if clean_retry is not None and clean_retry < clean_now:
            raise ValidationError("retry_at must not be earlier than now")
        with self._lock:
            job = self._require_current_lease(job_id, lease_token, clean_now)
            job.status = JobStatus.READY if clean_retry is not None else JobStatus.FAILED
            job.run_at = clean_retry if clean_retry is not None else job.run_at
            job.lease_owner = None
            job.lease_token = None
            job.lease_until = None
            return self._view(job)

    def inspect(self, job_id: str) -> JobView:
        clean_id = _clean_text(job_id, field="job_id")
        with self._lock:
            job = self._jobs.get(clean_id)
            if job is None:
                raise JobNotFound(f"job {clean_id!r} does not exist")
            return self._view(job)

    def snapshot(self) -> tuple[JobView, ...]:
        """Return stable state for metrics, debugging, or a recovery scan."""
        with self._lock:
            return tuple(self._view(job) for job in sorted(self._jobs.values(), key=lambda item: item.job_id))


def next_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Solution 1: bounded exponential retry delay."""
    clean_attempt = _positive_int(attempt, field="attempt")
    clean_base = _positive_float(base, field="base")
    clean_cap = _positive_float(cap, field="cap")
    if clean_base > clean_cap:
        raise ValidationError("base must not exceed cap")
    return min(clean_cap, clean_base * (2 ** (clean_attempt - 1)))


def is_due(run_at: float, now: float) -> bool:
    """Solution 2: compare finite schedule values at the polling boundary."""
    return _finite(run_at, field="run_at") <= _finite(now, field="now")


def _expect_error(callback: object, expected: type[BaseException]) -> None:
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        callback()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def run_self_checks() -> None:
    """Exercise ownership, expiry, retry, idempotency, and defensive copies."""
    assert next_backoff(1) == 1.0
    assert next_backoff(3) == 4.0
    assert next_backoff(99) == 60.0
    assert is_due(2.0, 2.0) is True
    assert is_due(3.0, 2.0) is False
    _expect_error(lambda: next_backoff(True), ValidationError)
    _expect_error(lambda: next_backoff(1, base=0), ValidationError)
    _expect_error(lambda: is_due(float("nan"), 0), ValidationError)

    clock = ManualClock()
    store = LeaseStore()
    payload = {"kind": "email", "recipient": "learner@example.test"}
    created = store.enqueue("job-1", payload)
    payload["recipient"] = "mutated@example.test"
    assert created.payload["recipient"] == "learner@example.test"
    assert store.inspect("job-1").payload["recipient"] == "learner@example.test"

    first = store.claim_due(worker_id="worker-a", now=clock.now(), lease_seconds=5)
    assert len(first) == 1
    old_lease = first[0]
    assert old_lease.attempt == 1
    assert store.claim_due(worker_id="worker-b", now=clock.now(), lease_seconds=5) == ()
    renewed = store.renew("job-1", old_lease.lease_token, now=1, lease_seconds=5)
    assert renewed.lease_until == 6.0
    clock.advance(5)
    assert store.claim_due(worker_id="worker-b", now=clock.now(), lease_seconds=5) == ()
    clock.advance(1)
    recovered = store.claim_due(worker_id="worker-b", now=clock.now(), lease_seconds=5)
    assert recovered[0].attempt == 2
    assert recovered[0].fencing_token > old_lease.fencing_token
    _expect_error(
        lambda: store.complete("job-1", old_lease.lease_token, now=clock.now()),
        LeaseConflict,
    )
    completed = store.complete("job-1", recovered[0].lease_token, now=clock.now())
    assert completed.status is JobStatus.COMPLETE
    _expect_error(
        lambda: store.complete("job-1", recovered[0].lease_token, now=clock.now()),
        LeaseConflict,
    )

    store.enqueue("retry-me", {"attempt": 0})
    retry_lease = store.claim_due(worker_id="worker-a", now=clock.now(), lease_seconds=3)[0]
    retry_view = store.fail(
        "retry-me",
        retry_lease.lease_token,
        now=clock.now(),
        retry_at=clock.now() + next_backoff(retry_lease.attempt),
    )
    assert retry_view.status is JobStatus.READY
    assert store.claim_due(worker_id="worker-a", now=clock.now(), lease_seconds=3) == ()
    clock.advance(1)
    retry_again = store.claim_due(worker_id="worker-a", now=clock.now(), lease_seconds=3)
    assert len(retry_again) == 1
    store.fail("retry-me", retry_again[0].lease_token, now=clock.now())

    _expect_error(
        lambda: store.enqueue("job-1", {"different": True}),
        JobAlreadyExists,
    )
    _expect_error(
        lambda: store.inspect("missing"),
        JobNotFound,
    )
    _expect_error(
        lambda: store.enqueue("", {}),
        ValidationError
    )


def main() -> None:
    """Show a deterministic scheduler recovery trace."""
    run_self_checks()

    clock = ManualClock()
    store = LeaseStore()
    store.enqueue("report-1", {"format": "json"})
    store.enqueue("report-2", {"format": "csv"}, run_at=10)

    first = store.claim_due(worker_id="worker-a", now=clock.now(), lease_seconds=4)
    print("Worker A claimed:", [(lease.job_id, lease.lease_until) for lease in first])

    clock.advance(4)
    recovered = store.claim_due(worker_id="worker-b", now=clock.now(), lease_seconds=4)
    print("Worker B recovered:", [(lease.job_id, lease.attempt) for lease in recovered])
    store.complete("report-1", recovered[0].lease_token, now=clock.now())

    clock.advance(6)
    scheduled = store.claim_due(worker_id="worker-b", now=clock.now(), lease_seconds=4)
    print("Scheduled work:", [lease.job_id for lease in scheduled])
    store.complete("report-2", scheduled[0].lease_token, now=clock.now())

    print("Final snapshot:", [(job.job_id, job.status.value, job.attempts) for job in store.snapshot()])
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
