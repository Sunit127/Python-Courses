"""Day 24: Asyncio event loops, tasks, and cancellation.

Learning goals
--------------
1. Explain how an event loop schedules cooperative coroutines.
2. Create tasks for independent work and preserve deterministic result order.
3. Bound concurrent async operations with a semaphore.
4. Apply timeouts, per-item failure handling, and cancellation correctly.
5. Shut down pending tasks without leaking work or swallowing cancellation.

Teaching notes
--------------
- An async function does not run when it is called; it returns a coroutine object.
  asyncio.run creates an event loop, drives the top-level coroutine, and closes
  the loop. Use one clear entry point for a command-line program.
- A task schedules a coroutine on the current loop. Awaiting a task suspends the
  current coroutine so other ready tasks can run. Async code is cooperative:
  a coroutine must await regularly; a CPU-heavy loop blocks every other task.
- asyncio.gather waits for several awaitables. By default it propagates the
  first exception; return_exceptions=True turns exceptions into result values.
  Prefer an explicit result type when a batch should report each item safely.
- A semaphore limits work that is active at once. It protects the remote service,
  database, or local resource, but it does not limit how many task objects you
  create. For unbounded streams, add a queue or a bounded task producer.
- asyncio.wait_for applies a timeout to one awaitable. A timeout cancels the
  underlying operation and raises TimeoutError to the caller. The operation must
  release resources in finally blocks.
- Cancellation is a control-flow signal, not an ordinary failure. CancelledError
  should normally propagate. Catch it only to clean up, then re-raise it. Do not
  use except Exception as a substitute for cancellation handling.
- When a parent operation is cancelled, cancel its child tasks, await them with
  return_exceptions=True, and then let the parent cancellation continue. This
  prevents orphaned tasks from running after the command has exited.
- Asyncio is excellent for many waiting operations. It does not make CPU-bound
  Python code parallel; move that work to a process pool and await the future.
  Never call blocking time.sleep, requests, or file operations in the event loop.

Run this file with Python 3.10+. It uses only the standard library, performs no
network access, and does not create persistent files.

Practice exercises
------------------
1. Add a retry policy to run_batch that retries only TimeoutError, with bounded
   exponential backoff and a final attempt count in each failure result.
2. Implement a producer-consumer version using asyncio.Queue. Keep at most a
   configured number of queued items and make every worker stop cleanly when the
   producer finishes.
3. Adapt a blocking CPU-bound function with loop.run_in_executor and compare its
   cancellation behavior with a native coroutine.

Solutions appear below the runnable example.

Expert challenge: resilient async job coordinator
--------------------------------------------------
Build an async coordinator for a multi-tenant notification service. It should
accept a large stream of validated jobs, enforce both global and per-tenant
concurrency limits, apply deadlines, retry only documented transient failures,
and stop accepting new work after cancellation. Emit structured outcomes in
input order while allowing completed results to be observed immediately.

Solution guidance
-----------------
1. Use a bounded asyncio.Queue per stage and a small, fixed worker set. A
   sentinel or TaskGroup-style lifecycle must let every worker exit.
2. Keep transport, retry policy, deadline calculation, and outcome formatting
   behind narrow interfaces. Inject the clock and sleeper for deterministic tests.
3. Use one semaphore for global capacity and a tenant-keyed semaphore for fairness.
   Acquire in a consistent order and release in finally blocks.
4. Give each job an immutable ID and deadline. Never retry validation, permission,
   or cancellation failures; use jitter only when it is safe and measurable.
5. On cancellation, stop the producer, cancel workers, drain or explicitly reject
   queued items, await every task, and return no misleading success result.
6. Test empty input, duplicate IDs, a timeout, a transient retry, permanent failure,
   tenant fairness, producer backpressure, cancellation during I/O, and shutdown
   with work still queued.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
import math


def _clean_text(value: str, *, field: str) -> str:
    """Return normalized non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field} cannot be empty")
    return cleaned


def _non_negative_float(value: float, *, field: str) -> float:
    """Return a finite non-negative number and reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _positive_int(value: int, *, field: str) -> int:
    """Return a positive integer and reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class WorkItem:
    """An immutable, pickle-free description of one async operation."""

    key: str
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _clean_text(self.key, field="key"))
        object.__setattr__(
            self,
            "delay_seconds",
            _non_negative_float(self.delay_seconds, field="delay_seconds"),
        )


@dataclass(frozen=True, slots=True)
class WorkSuccess:
    """A successful result tied to its input position."""

    index: int
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class WorkFailure:
    """A safe, structured description of one failed operation."""

    index: int
    key: str
    error_type: str
    message: str


WorkOutcome = WorkSuccess | WorkFailure
AsyncOperation = Callable[[WorkItem], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Immutable outcomes in the original input order."""

    outcomes: tuple[WorkOutcome, ...]

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.outcomes)))
        actual = tuple(outcome.index for outcome in self.outcomes)
        if actual != expected:
            raise ValueError("outcomes must be contiguous and ordered")

    @property
    def successes(self) -> tuple[WorkSuccess, ...]:
        """Return successful outcomes without exposing mutable state."""
        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, WorkSuccess)
        )

    @property
    def failures(self) -> tuple[WorkFailure, ...]:
        """Return failed outcomes without exposing mutable state."""
        return tuple(
            outcome
            for outcome in self.outcomes
            if isinstance(outcome, WorkFailure)
        )


def _safe_error_message(error: Exception) -> str:
    """Return bounded diagnostic text without tracebacks or secrets."""
    message = " ".join(str(error).split())
    return (message or "operation failed without a message")[:160]


async def simulated_operation(
    item: WorkItem,
    *,
    values: Mapping[str, str],
    failing_keys: frozenset[str] = frozenset(),
) -> str:
    """Simulate cooperative I/O without using a network or filesystem."""
    await asyncio.sleep(item.delay_seconds)
    if item.key in failing_keys:
        raise RuntimeError(f"simulated failure for {item.key!r}")
    try:
        return values[item.key]
    except KeyError as error:
        raise LookupError(f"unknown key: {item.key!r}") from error


async def _run_one(
    index: int,
    item: WorkItem,
    operation: AsyncOperation,
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
) -> WorkOutcome:
    """Run one operation while preserving cancellation semantics."""
    async with semaphore:
        try:
            value = await asyncio.wait_for(
                operation(item),
                timeout=timeout_seconds,
            )
            if not isinstance(value, str):
                raise TypeError("operation must return text")
            return WorkSuccess(index=index, key=item.key, value=value)
        except asyncio.TimeoutError:
            return WorkFailure(
                index=index,
                key=item.key,
                error_type="TimeoutError",
                message="operation exceeded its deadline",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return WorkFailure(
                index=index,
                key=item.key,
                error_type=type(error).__name__,
                message=_safe_error_message(error),
            )


async def run_batch(
    items: Iterable[WorkItem],
    operation: AsyncOperation,
    *,
    max_concurrency: int = 4,
    timeout_seconds: float = 1.0,
) -> BatchReport:
    """Run a finite async batch with bounded concurrency and ordered results."""
    concurrency = _positive_int(max_concurrency, field="max_concurrency")
    timeout = _non_negative_float(
        timeout_seconds,
        field="timeout_seconds",
    )
    if timeout == 0:
        raise ValueError("timeout_seconds must be positive")
    if not callable(operation):
        raise TypeError("operation must be callable")

    materialized = tuple(items)
    if any(not isinstance(item, WorkItem) for item in materialized):
        raise TypeError("items must contain WorkItem objects")
    if not materialized:
        return BatchReport(())

    semaphore = asyncio.Semaphore(min(concurrency, len(materialized)))
    tasks = [
        asyncio.create_task(
            _run_one(index, item, operation, semaphore, timeout),
            name=f"course-async-{index}",
        )
        for index, item in enumerate(materialized)
    ]
    try:
        outcomes = await asyncio.gather(*tasks)
    finally:
        # If the caller is cancelled, no child may outlive this batch.
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    return BatchReport(tuple(outcomes))


async def cancel_and_wait(task: asyncio.Task[object]) -> None:
    """Cancel a task and consume its expected cancellation signal."""
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def run_self_checks() -> None:
    """Verify order, limits, timeouts, failures, and cancellation."""
    values = {
        "slow": "slow result",
        "fast": "fast result",
        "also-fast": "another result",
    }
    items = (
        WorkItem("slow", delay_seconds=0.02),
        WorkItem("missing"),
        WorkItem("fast", delay_seconds=0.001),
    )
    report = await run_batch(
        items,
        lambda item: simulated_operation(item, values=values),
        max_concurrency=2,
        timeout_seconds=0.2,
    )
    assert tuple(outcome.key for outcome in report.outcomes) == (
        "slow",
        "missing",
        "fast",
    )
    assert tuple(outcome.index for outcome in report.successes) == (0, 2)
    assert report.failures[0].error_type == "LookupError"

    timeout_report = await run_batch(
        (WorkItem("slow", delay_seconds=0.03),),
        lambda item: simulated_operation(item, values=values),
        max_concurrency=1,
        timeout_seconds=0.001,
    )
    assert timeout_report.failures[0].error_type == "TimeoutError"

    started = asyncio.Event()

    async def never_finishes(item: WorkItem) -> str:
        del item
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    batch_task = asyncio.create_task(
        run_batch(
            (WorkItem("hang"),),
            never_finishes,
            max_concurrency=1,
            timeout_seconds=10,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    await cancel_and_wait(batch_task)
    assert batch_task.cancelled()

    for invalid_call in (
        lambda: WorkItem(""),
        lambda: WorkItem("bad", delay_seconds=-1),
    ):
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")

    async_invalid_calls = (
        lambda: run_batch(
            items,
            lambda item: simulated_operation(item, values=values),
            max_concurrency=0,
        ),
        lambda: run_batch(
            items,
            lambda item: simulated_operation(item, values=values),
            timeout_seconds=0,
        ),
        lambda: run_batch(
            (object(),),
            lambda item: simulated_operation(item, values=values),
        ),  # type: ignore[arg-type]
    )
    for invalid_call in async_invalid_calls:
        try:
            await invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid async input must raise a specific error")


async def main() -> None:
    """Run a deterministic local demonstration."""
    await run_self_checks()
    values = {
        "course-outline": "Asyncio schedules cooperative tasks.",
        "course-lesson": "Cancellation is part of the control flow.",
    }
    items = (
        WorkItem("course-outline", delay_seconds=0.01),
        WorkItem("missing"),
        WorkItem("course-lesson", delay_seconds=0.002),
    )
    report = await run_batch(
        items,
        lambda item: simulated_operation(item, values=values),
        max_concurrency=2,
        timeout_seconds=0.2,
    )
    for outcome in report.outcomes:
        if isinstance(outcome, WorkSuccess):
            print(f"OK   {outcome.key}: {outcome.value}")
        else:
            print(f"FAIL {outcome.key}: {outcome.error_type}: {outcome.message}")
    print("Self-checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
