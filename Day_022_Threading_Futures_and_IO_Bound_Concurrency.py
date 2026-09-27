"""Day 22: Threading, futures, and I/O-bound concurrency.

Learning goals
--------------
1. Distinguish I/O-bound waiting from CPU-bound computation.
2. Use ThreadPoolExecutor as a context-managed pool of worker threads.
3. Submit independent work, inspect Future objects, and collect completed tasks.
4. Preserve deterministic input order while tasks finish in any order.
5. Isolate expected per-task failures without hiding programmer errors.
6. Keep shared state small, protected, and replaceable with message passing.

Teaching notes
--------------
- Concurrency lets tasks make progress during overlapping periods. Parallelism
  means work literally runs at the same instant. Threads are often effective
  for blocking I/O because one thread can run while another waits.
- CPython's global interpreter lock does not make application state safe. It
  also means threads rarely speed up pure Python CPU-heavy work. Use measured
  evidence, and consider processes or native code for CPU-bound workloads.
- ThreadPoolExecutor owns worker threads. Use it as a context manager so normal
  completion and exceptions still shut the pool down cleanly.
- executor.map returns results in input order and raises worker exceptions when
  their result is reached. submit returns Future objects; as_completed yields
  futures in completion order and is useful for per-task error handling.
- Future.result() returns the worker's value or re-raises its exception in the
  coordinating thread. Catch Exception only where a documented batch policy
  allows one task to fail independently. Do not catch BaseException.
- Completion order is nondeterministic. Attach an input index or stable key to
  each future, then explicitly restore the order required by callers.
- Avoid sharing a database connection, open file, or mutable collection across
  workers unless its contract explicitly supports concurrent access. Prefer
  one resource per worker operation and protect unavoidable shared state with a
  lock.
- A lock protects a small critical section; it should not surround slow I/O.
  Holding a lock while waiting serializes the work and can cause deadlocks.
- cancel() only succeeds for work that has not started. Cooperative cancellation
  requires workers to check an explicit signal and leave resources consistent.
- Submitting an unbounded stream can consume unbounded memory. This lesson uses
  finite batches. A production pipeline should cap queued work and apply
  backpressure.
- Threads do not remove timeout, retry, idempotency, rate-limit, or security
  requirements. Concurrency makes those policies more important.

Run this file with Python 3.10+. It uses only the standard library, performs no
network access, and makes no persistent filesystem changes.

Practice exercises
------------------
1. Implement parallel_map(items, operation, max_workers). Preserve input order,
   support one-shot iterables, and reject invalid worker counts.
2. Implement summarize_batch(report). Return total, succeeded, failed, and
   successful character counts without mutating the report.
3. Implement failed_requests(requests, report). Return the original requests
   that failed, in input order, and validate that both inputs describe the same
   batch.

Solutions appear below the main concurrency example.

Expert challenge: bounded concurrent asset synchronizer
-------------------------------------------------------
Build a synchronizer that reads a manifest, fetches independent assets, verifies
their hashes, and atomically replaces destination files. Limit both total queued
work and per-host concurrency. Add explicit timeouts, a retry policy for only
documented transient failures, cooperative cancellation, structured results,
and metrics that never expose credentials.

Solution guidance
-----------------
1. Keep manifest validation, transport, hashing, and filesystem replacement in
   separate functions behind narrow interfaces.
2. Maintain at most a configured number of pending futures; submit a new item
   only after one completes to create backpressure.
3. Use one temporary sibling file per asset, flush and close it, verify the
   digest, then replace the destination atomically.
4. Make retries safe with immutable request IDs and cleanup of every temporary
   file. Never retry authentication or validation failures blindly.
5. Inject the transport, clock, sleeper, and cancellation signal so tests are
   deterministic and require no real network.
6. Test empty manifests, duplicate destinations, out-of-order completion,
   timeout, hash mismatch, partial failure, cancellation, and shutdown while
   work is pending.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from threading import Lock
import time
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


def _clean_text(value: str, *, field: str) -> str:
    """Return normalized, non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field} cannot be empty")
    return cleaned


def _positive_float(value: float, *, field: str) -> float:
    """Return a finite positive float, rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _positive_int(value: int, *, field: str) -> int:
    """Return a positive integer, rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """One finite, independently executable I/O request."""

    resource: str
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource",
            _clean_text(self.resource, field="resource"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_float(self.timeout_seconds, field="timeout_seconds"),
        )


@dataclass(frozen=True, slots=True)
class FetchSuccess:
    """A successful result tied to its original input position."""

    index: int
    resource: str
    character_count: int
    preview: str


@dataclass(frozen=True, slots=True)
class FetchFailure:
    """A safe, structured description of one failed request."""

    index: int
    resource: str
    error_type: str
    message: str


FetchOutcome = FetchSuccess | FetchFailure


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Immutable outcomes stored in original request order."""

    outcomes: tuple[FetchOutcome, ...]

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.outcomes)))
        actual = tuple(outcome.index for outcome in self.outcomes)
        if actual != expected:
            raise ValueError("outcomes must be contiguous and in input order")

    @property
    def successes(self) -> tuple[FetchSuccess, ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, FetchSuccess)
        )

    @property
    def failures(self) -> tuple[FetchFailure, ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, FetchFailure)
        )


FetchFunction = Callable[[str, float], str]


def _safe_error_message(error: Exception) -> str:
    """Return bounded diagnostic text suitable for this teaching example."""
    message = " ".join(str(error).split())
    if not message:
        message = "operation failed without a message"
    return message[:160]


def _fetch_one(
    index: int,
    request: FetchRequest,
    fetch: FetchFunction,
) -> FetchSuccess:
    """Execute one injected I/O operation inside a worker."""
    body = fetch(request.resource, request.timeout_seconds)
    if not isinstance(body, str):
        raise TypeError("fetch must return text")
    return FetchSuccess(
        index=index,
        resource=request.resource,
        character_count=len(body),
        preview=body[:24],
    )


def fetch_concurrently(
    requests: Iterable[FetchRequest],
    fetch: FetchFunction,
    *,
    max_workers: int = 4,
) -> BatchReport:
    """Run a finite I/O batch and convert worker failures into outcomes."""
    worker_count = _positive_int(max_workers, field="max_workers")
    if not callable(fetch):
        raise TypeError("fetch must be callable")

    batch = tuple(requests)
    for request in batch:
        if not isinstance(request, FetchRequest):
            raise TypeError("requests must contain FetchRequest objects")
    if not batch:
        return BatchReport(())

    outcomes: list[FetchOutcome | None] = [None] * len(batch)
    with ThreadPoolExecutor(
        max_workers=min(worker_count, len(batch)),
        thread_name_prefix="course-io",
    ) as executor:
        future_to_request: dict[
            Future[FetchSuccess],
            tuple[int, FetchRequest],
        ] = {
            executor.submit(_fetch_one, index, request, fetch): (index, request)
            for index, request in enumerate(batch)
        }

        for future in as_completed(future_to_request):
            index, request = future_to_request[future]
            try:
                outcomes[index] = future.result()
            except Exception as error:
                outcomes[index] = FetchFailure(
                    index=index,
                    resource=request.resource,
                    error_type=type(error).__name__,
                    message=_safe_error_message(error),
                )

    if any(outcome is None for outcome in outcomes):
        raise AssertionError("every submitted future must produce an outcome")
    return BatchReport(tuple(outcome for outcome in outcomes if outcome is not None))


class DemoFetcher:
    """Thread-safe in-memory stand-in for blocking I/O."""

    def __init__(
        self,
        pages: Mapping[str, str],
        *,
        delays: Mapping[str, float] | None = None,
    ) -> None:
        self._pages = dict(pages)
        self._delays = {
            key: _positive_float(value, field=f"delay for {key!r}")
            for key, value in (delays or {}).items()
        }
        self._calls: list[str] = []
        self._lock = Lock()

    def __call__(self, resource: str, timeout_seconds: float) -> str:
        resource = _clean_text(resource, field="resource")
        timeout = _positive_float(timeout_seconds, field="timeout_seconds")
        delay = self._delays.get(resource, 0.0)
        if delay > timeout:
            raise TimeoutError(f"request timed out for {resource!r}")

        if delay:
            time.sleep(delay)  # Simulated waiting; no lock is held.

        with self._lock:
            self._calls.append(resource)

        try:
            return self._pages[resource]
        except KeyError as error:
            raise LookupError(f"unknown resource: {resource!r}") from error

    @property
    def calls(self) -> tuple[str, ...]:
        """Return a defensive snapshot protected by the same lock as writes."""
        with self._lock:
            return tuple(self._calls)


def parallel_map(
    items: Iterable[T],
    operation: Callable[[T], R],
    *,
    max_workers: int,
) -> tuple[R, ...]:
    """Solution 1: transform a finite batch concurrently in input order."""
    worker_count = _positive_int(max_workers, field="max_workers")
    if not callable(operation):
        raise TypeError("operation must be callable")
    materialized = tuple(items)
    if not materialized:
        return ()

    with ThreadPoolExecutor(
        max_workers=min(worker_count, len(materialized)),
        thread_name_prefix="course-map",
    ) as executor:
        return tuple(executor.map(operation, materialized))


def summarize_batch(report: BatchReport) -> dict[str, int]:
    """Solution 2: calculate a stable summary of immutable outcomes."""
    if not isinstance(report, BatchReport):
        raise TypeError("report must be a BatchReport")
    return {
        "total": len(report.outcomes),
        "succeeded": len(report.successes),
        "failed": len(report.failures),
        "characters": sum(
            outcome.character_count for outcome in report.successes
        ),
    }


def failed_requests(
    requests: Iterable[FetchRequest],
    report: BatchReport,
) -> tuple[FetchRequest, ...]:
    """Solution 3: recover failed inputs without depending on completion order."""
    if not isinstance(report, BatchReport):
        raise TypeError("report must be a BatchReport")
    batch = tuple(requests)
    if len(batch) != len(report.outcomes):
        raise ValueError("requests and report must describe the same batch")

    failed_indices = {failure.index for failure in report.failures}
    return tuple(
        request
        for index, request in enumerate(batch)
        if index in failed_indices
    )


def run_self_checks() -> None:
    """Verify ordering, error isolation, validation, and thread-safe state."""
    requests = (
        FetchRequest("slow", timeout_seconds=0.2),
        FetchRequest("missing"),
        FetchRequest("fast"),
    )
    fetcher = DemoFetcher(
        {"slow": "slow response", "fast": "fast response"},
        delays={"slow": 0.02, "fast": 0.001},
    )
    report = fetch_concurrently(requests, fetcher, max_workers=3)

    assert tuple(outcome.resource for outcome in report.outcomes) == (
        "slow",
        "missing",
        "fast",
    )
    assert tuple(success.index for success in report.successes) == (0, 2)
    assert report.failures[0].index == 1
    assert report.failures[0].error_type == "LookupError"
    assert summarize_batch(report) == {
        "total": 3,
        "succeeded": 2,
        "failed": 1,
        "characters": 26,
    }
    assert failed_requests(requests, report) == (requests[1],)
    assert sorted(fetcher.calls) == ["fast", "missing", "slow"]

    assert parallel_map((value for value in range(5)), lambda value: value**2, max_workers=2) == (
        0,
        1,
        4,
        9,
        16,
    )
    assert fetch_concurrently((), fetcher, max_workers=1) == BatchReport(())

    bad_return_report = fetch_concurrently(
        (FetchRequest("binary"),),
        lambda resource, timeout: b"not text",  # type: ignore[return-value]
        max_workers=1,
    )
    assert bad_return_report.failures[0].error_type == "TypeError"

    timeout_report = fetch_concurrently(
        (FetchRequest("too-slow", timeout_seconds=0.001),),
        DemoFetcher(
            {"too-slow": "late"},
            delays={"too-slow": 0.01},
        ),
        max_workers=1,
    )
    assert timeout_report.failures[0].error_type == "TimeoutError"

    invalid_calls = (
        lambda: FetchRequest(""),
        lambda: FetchRequest("page", timeout_seconds=float("inf")),
        lambda: fetch_concurrently((object(),), fetcher),  # type: ignore[arg-type]
        lambda: fetch_concurrently(requests, fetcher, max_workers=True),
        lambda: parallel_map((1,), 42, max_workers=1),  # type: ignore[arg-type]
        lambda: failed_requests(requests[:-1], report),
        lambda: BatchReport(
            (
                FetchFailure(1, "wrong", "ValueError", "wrong index"),
            )
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a deterministic local demonstration with no external I/O."""
    run_self_checks()
    requests = (
        FetchRequest("course://outline"),
        FetchRequest("course://missing"),
        FetchRequest("course://lesson"),
    )
    fetcher = DemoFetcher(
        {
            "course://outline": "90-day Python journey",
            "course://lesson": "Thread pools overlap waiting work.",
        },
        delays={
            "course://outline": 0.01,
            "course://lesson": 0.002,
        },
    )
    report = fetch_concurrently(requests, fetcher, max_workers=3)

    for outcome in report.outcomes:
        if isinstance(outcome, FetchSuccess):
            print(
                f"OK   {outcome.resource}: "
                f"{outcome.character_count} characters"
            )
        else:
            print(
                f"FAIL {outcome.resource}: "
                f"{outcome.error_type}: {outcome.message}"
            )
    print("Summary:", summarize_batch(report))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
