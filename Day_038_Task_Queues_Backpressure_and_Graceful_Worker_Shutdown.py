"""Day 38: Task queues, backpressure, and graceful worker shutdown.

Learning goals
--------------
1. Separate task admission, buffering, execution, and shutdown responsibilities.
2. Bound queue capacity so overload becomes visible instead of consuming memory.
3. Apply backpressure with explicit accept, reject, and closed outcomes.
4. Shut workers down by closing ingress, draining accepted work, and joining threads.
5. Test concurrency invariants without depending on arbitrary sleep durations.

Teaching notes
--------------
- A task queue decouples producers from workers, but it does not create capacity.
  If producers remain faster than consumers, an unbounded queue only delays failure
  while increasing memory use and request latency.
- Backpressure is a contract for overload. Common policies include blocking the
  producer, rejecting work, shedding low-priority work, or scaling consumers.
  Choose deliberately and expose the outcome to callers and metrics.
- Queue capacity should come from a latency and burst budget, then be validated
  with measurements. A huge buffer can make a service look available while tasks
  wait too long to remain useful.
- Admission and execution are different states. Once this lesson returns ACCEPTED,
  the pool owns the task and graceful shutdown must either finish it or report it
  as discarded. Durable systems must persist accepted work before acknowledging it.
- Graceful shutdown happens in phases: stop admission, drain already accepted work,
  stop worker loops, and join worker threads. Immediate shutdown instead finishes
  work already running and explicitly discards work that has not started.
- queue.Queue provides bounded, thread-safe put/get operations and unfinished-task
  accounting. Every successful get must have exactly one matching task_done call,
  including when a handler raises, or join can wait forever.
- Timeouts are useful for periodically checking lifecycle state. A timeout is not
  task cancellation: Python cannot safely kill a running thread. Long handlers need
  their own deadlines or cooperative cancellation tokens.
- Worker failures must not silently kill a worker. Record a bounded, non-secret
  diagnostic and continue according to an explicit retry or dead-letter policy.
- Avoid tests that merely sleep and hope. Events, joins, bounded waits, and exact
  counters make concurrency tests faster and less flaky.
- This in-memory example models one process. Production queues also need durable
  acknowledgement, visibility timeouts, leases, retry limits, deduplication,
  observability, deployment coordination, and crash recovery.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, writes no files,
and completes without arbitrary sleeps.

Practice exercises
------------------
1. Implement load_factor(queued, capacity). Return a value from 0.0 through 1.0
   and reject Booleans, negative counts, zero capacity, or queued above capacity.
2. Implement should_shed(load, threshold=0.8). Reject non-finite numbers and
   return True when load is greater than or equal to the threshold.
3. Implement submit_batch(task_queue, tasks). Preserve input order, attempt every
   task, and return an immutable result for each admission attempt.

Solutions appear in the executable code below.

Expert challenge: production job service
----------------------------------------
Build a durable, multi-tenant job service. Persist a job before acknowledging it,
lease jobs to workers with an expiry, and make completion idempotent. Add bounded
retries with jitter, a dead-letter workflow, per-tenant quotas, priority aging so
low-priority jobs cannot starve, and cancellation deadlines. Expose queue depth,
oldest-job age, admission rejections, execution latency, failures, and shutdown
duration. Test worker crashes, lease expiry, duplicate completion, rolling deploys,
poison jobs, overload, and recovery without losing accepted work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from types import MappingProxyType
from typing import Protocol


class QueueLifecycleError(RuntimeError):
    """Raised when an operation violates the queue or pool lifecycle."""


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _clean_text(value: object, *, field: str, maximum: int = 120) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters")
    return cleaned


def _safe_error(error: Exception) -> str:
    """Return a bounded diagnostic without traceback or argument representations."""
    message = " ".join(str(error).split())
    name = type(error).__name__
    return f"{name}: {message}"[:240] if message else name


@dataclass(frozen=True, slots=True)
class Task:
    """An immutable task envelope with defensively copied metadata."""

    task_id: str
    kind: str
    _payload: Mapping[str, object]

    def __init__(
        self,
        task_id: str,
        kind: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping or None")
        copied = dict(payload or {})
        if any(not isinstance(key, str) for key in copied):
            raise TypeError("payload keys must be text")
        object.__setattr__(self, "task_id", _clean_text(task_id, field="task_id"))
        object.__setattr__(self, "kind", _clean_text(kind, field="kind"))
        object.__setattr__(self, "_payload", MappingProxyType(copied))

    @property
    def payload(self) -> dict[str, object]:
        """Return a copy so callers cannot mutate the accepted task."""
        return dict(self._payload)


class Admission(Enum):
    """The complete set of producer-visible admission outcomes."""

    ACCEPTED = "accepted"
    OVERLOADED = "overloaded"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """One task's admission decision."""

    task_id: str
    outcome: Admission

    @property
    def accepted(self) -> bool:
        return self.outcome is Admission.ACCEPTED


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """A consistent point-in-time view of bounded queue counters."""

    capacity: int
    queued: int
    unfinished: int
    accepted: int
    rejected_overload: int
    rejected_closed: int
    closed: bool

    @property
    def load(self) -> float:
        return load_factor(self.queued, self.capacity)


class BoundedTaskQueue:
    """Thread-safe bounded admission with explicit overload and closure."""

    def __init__(self, capacity: int) -> None:
        self._capacity = _positive_int(capacity, field="capacity")
        self._queue: Queue[Task] = Queue(maxsize=self._capacity)
        self._lock = Lock()
        self._closed = False
        self._accepted = 0
        self._rejected_overload = 0
        self._rejected_closed = 0

    def submit(self, task: Task) -> AdmissionResult:
        """Accept immediately or report pressure; never block a producer."""
        if not isinstance(task, Task):
            raise TypeError("task must be Task")
        with self._lock:
            if self._closed:
                self._rejected_closed += 1
                return AdmissionResult(task.task_id, Admission.CLOSED)
            try:
                self._queue.put_nowait(task)
            except Full:
                self._rejected_overload += 1
                return AdmissionResult(task.task_id, Admission.OVERLOADED)
            self._accepted += 1
            return AdmissionResult(task.task_id, Admission.ACCEPTED)

    def close(self) -> bool:
        """Stop future admission; return whether this call changed state."""
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            return True

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def take(self, timeout: float) -> Task | None:
        """Return one task or None after a bounded wait."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        try:
            return self._queue.get(timeout=float(timeout))
        except Empty:
            return None

    def complete(self) -> None:
        """Match one successful take call."""
        self._queue.task_done()

    def wait_until_idle(self) -> None:
        """Wait until every accepted, non-discarded task is marked complete."""
        self._queue.join()

    def discard_pending(self) -> tuple[Task, ...]:
        """Remove tasks that have not started and balance unfinished accounting."""
        discarded: list[Task] = []
        while True:
            try:
                task = self._queue.get_nowait()
            except Empty:
                return tuple(discarded)
            discarded.append(task)
            self._queue.task_done()

    def snapshot(self) -> QueueSnapshot:
        with self._lock:
            return QueueSnapshot(
                capacity=self._capacity,
                queued=self._queue.qsize(),
                unfinished=self._queue.unfinished_tasks,
                accepted=self._accepted,
                rejected_overload=self._rejected_overload,
                rejected_closed=self._rejected_closed,
                closed=self._closed,
            )


class TaskHandler(Protocol):
    """A callable that completes one task or raises a classified failure."""

    def __call__(self, task: Task) -> None:
        ...


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """A safe, bounded record of a handler failure."""

    task_id: str
    worker_name: str
    detail: str


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    """Worker-pool results after any point in its lifecycle."""

    started: bool
    stopping: bool
    processed: int
    failed: int
    discarded: int
    failures: tuple[FailureRecord, ...]


class WorkerPool:
    """A fixed-size worker pool with graceful and immediate shutdown modes."""

    def __init__(
        self,
        task_queue: BoundedTaskQueue,
        handler: TaskHandler,
        *,
        workers: int = 2,
        poll_interval: float = 0.05,
    ) -> None:
        if not isinstance(task_queue, BoundedTaskQueue):
            raise TypeError("task_queue must be BoundedTaskQueue")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._queue = task_queue
        self._handler = handler
        self._worker_count = _positive_int(workers, field="workers")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
        ):
            raise TypeError("poll_interval must be a number")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be a positive finite number")
        self._poll_interval = float(poll_interval)
        self._stop_now = Event()
        self._lock = Lock()
        self._threads: list[Thread] = []
        self._started = False
        self._stopping = False
        self._processed = 0
        self._discarded = 0
        self._failures: list[FailureRecord] = []

    def start(self) -> None:
        """Start workers exactly once."""
        with self._lock:
            if self._started:
                raise QueueLifecycleError("worker pool has already started")
            if self._queue.closed:
                raise QueueLifecycleError("cannot start with closed ingress")
            self._started = True
            threads = [
                Thread(
                    target=self._worker_loop,
                    name=f"task-worker-{index}",
                    daemon=False,
                )
                for index in range(1, self._worker_count + 1)
            ]
            self._threads = threads
        for thread in threads:
            thread.start()

    def _worker_loop(self) -> None:
        from threading import current_thread

        worker_name = current_thread().name
        while not self._stop_now.is_set():
            task = self._queue.take(self._poll_interval)
            if task is None:
                snapshot = self._queue.snapshot()
                if snapshot.closed and snapshot.unfinished == 0:
                    return
                continue
            try:
                self._handler(task)
            except Exception as error:
                failure = FailureRecord(
                    task_id=task.task_id,
                    worker_name=worker_name,
                    detail=_safe_error(error),
                )
                with self._lock:
                    self._failures.append(failure)
            else:
                with self._lock:
                    self._processed += 1
            finally:
                self._queue.complete()

    def shutdown(self, *, drain: bool = True) -> tuple[Task, ...]:
        """Close ingress, optionally drain, stop loops, and join every worker."""
        if not isinstance(drain, bool):
            raise TypeError("drain must be a Boolean")
        with self._lock:
            if not self._started:
                raise QueueLifecycleError("worker pool has not started")
            if self._stopping:
                raise QueueLifecycleError("worker pool is already stopping")
            self._stopping = True

        self._queue.close()
        discarded: tuple[Task, ...] = ()
        if drain:
            self._queue.wait_until_idle()
        else:
            self._stop_now.set()
            discarded = self._queue.discard_pending()
            with self._lock:
                self._discarded += len(discarded)

        self._stop_now.set()
        for thread in self._threads:
            thread.join()
        return discarded

    def snapshot(self) -> PoolSnapshot:
        with self._lock:
            return PoolSnapshot(
                started=self._started,
                stopping=self._stopping,
                processed=self._processed,
                failed=len(self._failures),
                discarded=self._discarded,
                failures=tuple(self._failures),
            )


def load_factor(queued: int, capacity: int) -> float:
    """Solution 1: calculate bounded queue occupancy."""
    count = _non_negative_int(queued, field="queued")
    limit = _positive_int(capacity, field="capacity")
    if count > limit:
        raise ValueError("queued must not exceed capacity")
    return count / limit


def should_shed(load: float, threshold: float = 0.8) -> bool:
    """Solution 2: decide when optional work should be rejected early."""
    for value, field in ((load, "load"), (threshold, "threshold")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be between 0.0 and 1.0")
    return load >= threshold


def submit_batch(
    task_queue: BoundedTaskQueue,
    tasks: Iterable[Task],
) -> tuple[AdmissionResult, ...]:
    """Solution 3: attempt each admission without hiding partial acceptance."""
    if not isinstance(task_queue, BoundedTaskQueue):
        raise TypeError("task_queue must be BoundedTaskQueue")
    try:
        iterator = iter(tasks)
    except TypeError as error:
        raise TypeError("tasks must be iterable") from error

    results: list[AdmissionResult] = []
    for task in iterator:
        results.append(task_queue.submit(task))
    return tuple(results)


class RecordingHandler:
    """A synchronized example handler controlled with events, not sleeps."""

    def __init__(
        self,
        *,
        release: Event | None = None,
        fail_ids: Iterable[str] = (),
    ) -> None:
        self.started = Event()
        self.release = release
        self._fail_ids = {_clean_text(value, field="fail_id") for value in fail_ids}
        self._lock = Lock()
        self._handled: list[str] = []

    def __call__(self, task: Task) -> None:
        self.started.set()
        if self.release is not None and not self.release.wait(timeout=2.0):
            raise TimeoutError("test handler was not released")
        if task.task_id in self._fail_ids:
            raise RuntimeError("simulated handler failure")
        with self._lock:
            self._handled.append(task.task_id)

    @property
    def handled(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._handled)


def run_self_checks() -> None:
    """Verify admission, failure isolation, draining, and immediate shutdown."""
    assert load_factor(0, 4) == 0.0
    assert load_factor(3, 4) == 0.75
    assert load_factor(4, 4) == 1.0
    assert should_shed(0.79) is False
    assert should_shed(0.8) is True

    copied_source: dict[str, object] = {"value": 1}
    copied_task = Task("copy", "demo", copied_source)
    copied_source["value"] = 99
    detached = copied_task.payload
    detached["value"] = 100
    assert copied_task.payload == {"value": 1}

    admission_queue = BoundedTaskQueue(capacity=2)
    batch = submit_batch(
        admission_queue,
        (
            Task("a", "demo"),
            Task("b", "demo"),
            Task("c", "demo"),
        ),
    )
    assert tuple(result.outcome for result in batch) == (
        Admission.ACCEPTED,
        Admission.ACCEPTED,
        Admission.OVERLOADED,
    )
    assert admission_queue.snapshot().load == 1.0
    assert admission_queue.close() is True
    assert admission_queue.close() is False
    assert (
        admission_queue.submit(Task("d", "demo")).outcome
        is Admission.CLOSED
    )
    assert tuple(task.task_id for task in admission_queue.discard_pending()) == (
        "a",
        "b",
    )
    admission_queue.wait_until_idle()

    graceful_queue = BoundedTaskQueue(capacity=4)
    graceful_handler = RecordingHandler(fail_ids={"bad"})
    graceful_pool = WorkerPool(graceful_queue, graceful_handler, workers=2)
    graceful_pool.start()
    graceful_results = submit_batch(
        graceful_queue,
        (
            Task("one", "demo"),
            Task("bad", "demo"),
            Task("two", "demo"),
        ),
    )
    assert all(result.accepted for result in graceful_results)
    assert graceful_pool.shutdown(drain=True) == ()
    graceful_snapshot = graceful_pool.snapshot()
    assert graceful_snapshot.processed == 2
    assert graceful_snapshot.failed == 1
    assert graceful_snapshot.discarded == 0
    assert len(graceful_snapshot.failures) == 1
    assert graceful_snapshot.failures[0].task_id == "bad"
    assert set(graceful_handler.handled) == {"one", "two"}
    assert graceful_queue.snapshot().unfinished == 0
    assert (
        graceful_queue.submit(Task("late", "demo")).outcome
        is Admission.CLOSED
    )

    release = Event()
    immediate_queue = BoundedTaskQueue(capacity=4)
    immediate_handler = RecordingHandler(release=release)
    immediate_pool = WorkerPool(immediate_queue, immediate_handler, workers=1)
    immediate_pool.start()
    assert immediate_queue.submit(Task("running", "demo")).accepted
    assert immediate_handler.started.wait(timeout=1.0)
    assert immediate_queue.submit(Task("pending-1", "demo")).accepted
    assert immediate_queue.submit(Task("pending-2", "demo")).accepted

    discarded_holder: list[tuple[Task, ...]] = []

    def stop_pool() -> None:
        discarded_holder.append(immediate_pool.shutdown(drain=False))

    stopper = Thread(target=stop_pool, name="shutdown-coordinator")
    stopper.start()
    release.set()
    stopper.join(timeout=2.0)
    assert not stopper.is_alive()
    discarded_ids = tuple(task.task_id for task in discarded_holder[0])
    assert discarded_ids == ("pending-1", "pending-2")
    immediate_snapshot = immediate_pool.snapshot()
    assert immediate_snapshot.processed == 1
    assert immediate_snapshot.failed == 0
    assert immediate_snapshot.discarded == 2
    assert immediate_handler.handled == ("running",)
    assert immediate_queue.snapshot().unfinished == 0

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: BoundedTaskQueue(0),
        lambda: BoundedTaskQueue(True),
        lambda: Task("", "demo"),
        lambda: Task("id", "", {}),
        lambda: Task("id", "demo", {1: "bad"}),  # type: ignore[dict-item]
        lambda: load_factor(-1, 2),
        lambda: load_factor(3, 2),
        lambda: should_shed(float("nan")),
        lambda: should_shed(0.5, 1.1),
        lambda: admission_queue.take(0),
        lambda: WorkerPool(BoundedTaskQueue(1), lambda task: None, workers=0),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError, QueueLifecycleError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a small, fully drained worker-pool demonstration."""
    run_self_checks()

    task_queue = BoundedTaskQueue(capacity=3)
    handler = RecordingHandler()
    pool = WorkerPool(task_queue, handler, workers=2)
    pool.start()

    results = submit_batch(
        task_queue,
        (
            Task("report-001", "build-report", {"format": "json"}),
            Task("report-002", "build-report", {"format": "text"}),
            Task("report-003", "build-report", {"format": "csv"}),
        ),
    )
    pool.shutdown(drain=True)

    print("Admission:", [result.outcome.value for result in results])
    print("Handled:", sorted(handler.handled))
    print("Queue closed:", task_queue.snapshot().closed)
    print("Pool snapshot:", pool.snapshot())
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
