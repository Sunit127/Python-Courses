"""Day 23: Multiprocessing, process pools, and CPU-bound parallelism.

Learning goals
--------------
1. Distinguish CPU-bound parallelism from I/O-bound concurrency.
2. Use ProcessPoolExecutor with top-level, pickleable worker functions.
3. Partition independent work and preserve deterministic result ordering.
4. Validate process-pool inputs and choose a useful worker count.
5. Explain process isolation, startup cost, chunksize, and the main guard.

Teaching notes
--------------
- Threads are a good fit for waiting on I/O. A CPU-bound pure-Python function
  usually needs processes (or native/vectorized code) to run on multiple cores.
  Measure first: process startup, serialization, and data transfer are overhead.
- A process has its own interpreter and memory. Mutating a global list in a child
  does not mutate the parent's list. Return values and exceptions cross the
  boundary through pickle serialization.
- ProcessPoolExecutor submits work to worker processes. Functions and arguments
  must be importable and pickleable; nested functions, open sockets, locks, and
  many live objects are not valid payloads.
- Keep worker functions at module scope and protect process-pool startup with
  if __name__ == "__main__":. This is essential for spawn-based platforms, where
  a child imports the module again.
- executor.map returns values in input order even when jobs finish out of order.
  ProcessPoolExecutor.map also accepts chunksize: larger chunks can reduce
  scheduling overhead for many tiny jobs, while smaller chunks improve balance.
- Do not create an unbounded number of futures or pass a giant object to every
  worker. Partition the data, send only what each worker needs, and bound work
  in flight when a stream is larger than memory.
- A pool is not a cancellation or retry policy. Handle timeouts, failures,
  idempotency, and shutdown deliberately. Never retry arbitrary side effects.
- CPU parallelism can make observability harder: include stable job IDs in
  results, keep logs structured, and avoid having every process write to the
  same file or database connection.

Run this file with Python 3.10+. It uses only the standard library, performs no
network access, and does not create persistent files.

Practice exercises
------------------
1. Implement count_primes_up_to(limit) as a single-process baseline. Compare its
   answer with count_primes_up_to_parallel(limit, parts, max_workers).
2. Extend parallel_count_ranges so it returns (job, count) pairs and verify that
   the jobs remain in input order even if one range takes longer.
3. Design a bounded process-pool consumer for a very large iterable. Explain how
   you would limit pending work, propagate the first failure, and shut down the
   pool without leaking resources.

Solutions appear below the runnable example.

Expert challenge: parallel checksum index
-----------------------------------------
Build a command-line tool that scans a manifest of immutable, local data chunks,
computes a CPU-heavy checksum for each chunk in parallel, and writes a deterministic
index. Add a sequential baseline, measured speedup, configurable worker count,
bounded submission, stable job IDs, atomic output replacement, and a resume mode
that skips chunks whose size and digest already match. The tool must never execute
untrusted input as code and must report failures without leaking secret paths.

Solution guidance
-----------------
1. Keep manifest parsing, chunk reading, checksum calculation, and index writing
   in separate functions behind narrow interfaces.
2. Send a small immutable job record (ID, path, expected size) to a top-level
   worker. Open the file inside that worker and close it with a context manager.
3. Submit at most a fixed number of futures; as each completes, submit the next
   job. Store results by job ID, then sort IDs before serializing the index.
4. Use a temporary file in the destination directory, flush it, optionally
   fsync it, and replace the final index atomically only after every required
   result succeeds.
5. Test empty input, duplicate IDs, missing files, checksum mismatches, a worker
   exception, cancellation, partial output, and a worker count larger than the
   number of jobs. Benchmark against the single-process baseline rather than
   assuming more processes are faster.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math


def _positive_int(value: int, *, field: str) -> int:
    """Return a positive integer and reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _non_negative_int(value: int, *, field: str) -> int:
    """Return a non-negative integer and reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def is_prime(number: int) -> bool:
    """Return whether a non-negative integer is prime."""
    _non_negative_int(number, field="number")
    if number < 2:
        return False
    if number % 2 == 0:
        return number == 2
    divisor = 3
    limit = math.isqrt(number)
    while divisor <= limit:
        if number % divisor == 0:
            return False
        divisor += 2
    return True


def count_primes(limit: int) -> int:
    """Count primes in the half-open interval [0, limit)."""
    _non_negative_int(limit, field="limit")
    return sum(1 for candidate in range(limit) if is_prime(candidate))


@dataclass(frozen=True, slots=True)
class PrimeRange:
    """An immutable, pickleable half-open interval [start, stop)."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _non_negative_int(self.start, field="start")
        _non_negative_int(self.stop, field="stop")
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start")

    @property
    def size(self) -> int:
        """Return the number of candidate integers in this range."""
        return self.stop - self.start


def count_primes_in_range(job: PrimeRange) -> int:
    """Top-level worker: count primes in one independent range."""
    if not isinstance(job, PrimeRange):
        raise TypeError("job must be a PrimeRange")
    return sum(
        1
        for candidate in range(job.start, job.stop)
        if is_prime(candidate)
    )


def partition_interval(limit: int, parts: int) -> tuple[PrimeRange, ...]:
    """Partition [2, limit + 1) into non-empty, nearly even ranges."""
    _non_negative_int(limit, field="limit")
    part_count = _positive_int(parts, field="parts")
    if limit < 2:
        return ()

    candidate_count = limit - 1  # candidates are 2 through limit inclusive
    actual_parts = min(part_count, candidate_count)
    base, remainder = divmod(candidate_count, actual_parts)
    ranges: list[PrimeRange] = []
    start = 2
    for index in range(actual_parts):
        width = base + (1 if index < remainder else 0)
        stop = start + width
        ranges.append(PrimeRange(start, stop))
        start = stop
    return tuple(ranges)


def parallel_count_ranges(
    jobs: Iterable[PrimeRange],
    *,
    max_workers: int = 2,
) -> tuple[int, ...]:
    """Count each range in processes and return results in input order."""
    worker_count = _positive_int(max_workers, field="max_workers")
    materialized = tuple(jobs)
    if any(not isinstance(job, PrimeRange) for job in materialized):
        raise TypeError("jobs must contain PrimeRange objects")
    if not materialized:
        return ()

    with ProcessPoolExecutor(
        max_workers=min(worker_count, len(materialized)),
    ) as executor:
        # chunksize is explicit so readers can reason about batching overhead.
        return tuple(executor.map(count_primes_in_range, materialized, chunksize=1))


def count_primes_up_to_parallel(
    limit: int,
    *,
    parts: int = 4,
    max_workers: int = 2,
) -> int:
    """Count primes through limit by summing independent process results."""
    jobs = partition_interval(limit, parts)
    return sum(parallel_count_ranges(jobs, max_workers=max_workers))


def run_self_checks() -> None:
    """Verify answers, partition coverage, ordering, and input validation."""
    assert is_prime(2)
    assert is_prime(97)
    assert not is_prime(0)
    assert not is_prime(1)
    assert not is_prime(100)
    assert count_primes(0) == 0
    assert count_primes(10) == 4  # 2, 3, 5, 7
    assert count_primes(30) == 10

    jobs = partition_interval(30, parts=3)
    assert jobs[0].start == 2
    assert jobs[-1].stop == 31
    assert sum(job.size for job in jobs) == 29
    counts = parallel_count_ranges(jobs, max_workers=2)
    assert sum(counts) == count_primes(30)
    assert counts == (5, 3, 2)
    assert count_primes_up_to_parallel(100, parts=7, max_workers=3) == 25
    assert partition_interval(1, parts=3) == ()
    assert parallel_count_ranges((), max_workers=1) == ()

    invalid_calls = (
        lambda: is_prime(True),
        lambda: count_primes(-1),
        lambda: PrimeRange(5, 5),
        lambda: partition_interval(10, parts=0),
        lambda: parallel_count_ranges((object(),)),  # type: ignore[arg-type]
        lambda: parallel_count_ranges(jobs, max_workers=False),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a deterministic local demonstration."""
    run_self_checks()
    jobs = partition_interval(120, parts=4)
    counts = parallel_count_ranges(jobs, max_workers=2)
    print("Ranges:", [(job.start, job.stop) for job in jobs])
    print("Prime counts:", counts)
    print("Total primes through 120:", sum(counts))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
