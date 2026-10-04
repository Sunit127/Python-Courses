"""Day 29: Performance measurement, profiling, and optimization.

Learning goals
--------------
1. Measure elapsed time with a monotonic clock instead of guessing.
2. Use repeatable benchmarks and compare distributions, not one lucky run.
3. Profile a real call to find where CPU time is spent.
4. Apply caching and data-oriented changes while preserving behavior.
5. Set a performance budget and guard it with deterministic tests.

Teaching notes
--------------
- Optimize after measuring. A profiler identifies hot paths; it does not prove
  that a change is safe or useful for every workload.
- Use time.perf_counter() for elapsed durations. It is monotonic and has high
  resolution; datetime.now() is for timestamps, not benchmarks.
- One timing includes setup, scheduling noise, and garbage collection. Run
  several repeats, report a stable statistic such as the minimum and average,
  and compare equivalent inputs.
- timeit is convenient for tiny expressions. For application code, a small
  benchmark function with injected clock and representative data is easier to
  test and review.
- cProfile measures call counts and cumulative time. Sort by cumulative time to
  find expensive call chains, then inspect a focused function before changing it.
- functools.lru_cache trades memory for speed. Cache only deterministic
  functions, choose a bounded size when inputs can grow, and include every
  result-changing argument in the cache key.
- Avoid accidental quadratic work: repeated list membership scans, string
  concatenation in a loop, and sorting the same data repeatedly are common
  examples. A set, join, or one-time sort can change the growth rate.
- A faster algorithm is not automatically a better API. Keep validation and
  output semantics identical, document the workload assumption, and rerun
  correctness tests after every optimization.
- Benchmark thresholds are environment-dependent. Use generous budgets in CI,
  compare against a baseline, and investigate regressions instead of making
  flaky exact-time assertions.

Run this file with Python 3.10+ to execute deterministic examples and checks.
It uses only the standard library and performs no network or persistent I/O.

Practice exercises
------------------
1. Implement time_callable(function, *args, clock=...). Return the function's
   result and elapsed seconds, rejecting a non-callable function.
2. Implement cached_fibonacci(n) with a bounded cache and input validation.
3. Implement benchmark_many(named_functions, argument), preserving mapping
   order and returning one BenchmarkResult per function.

Solutions appear below the main examples.

Expert challenge: performance regression guard
----------------------------------------------
Build a small regression guard for a text-processing pipeline. Benchmark a
baseline and candidate on the same immutable input, verify identical output,
and fail only when the candidate exceeds a configurable relative budget.
Include a profile report for a failing candidate and keep timing, profiling,
and business logic separate. In production, store historical results by
interpreter, hardware, and input size instead of comparing unrelated machines.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import cProfile
import io
import math
import pstats
import re
import time
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")
Clock = Callable[[], float]

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Summary of repeated elapsed-time measurements in seconds."""

    runs: int
    minimum_seconds: float
    average_seconds: float

    def __post_init__(self) -> None:
        _positive_int(self.runs, "runs")
        _non_negative_float(self.minimum_seconds, "minimum_seconds")
        _non_negative_float(self.average_seconds, "average_seconds")
        if self.minimum_seconds > self.average_seconds:
            raise ValueError("minimum cannot exceed average")


def benchmark(
    function: Callable[..., R],
    *args: object,
    repeats: int = 5,
    clock: Clock = time.perf_counter,
    **kwargs: object,
) -> BenchmarkResult:
    """Run one callable several times and summarize elapsed durations."""
    if not callable(function):
        raise TypeError("function must be callable")
    count = _positive_int(repeats, "repeats")
    if not callable(clock):
        raise TypeError("clock must be callable")

    durations: list[float] = []
    for _ in range(count):
        started = clock()
        function(*args, **kwargs)
        elapsed = clock() - started
        durations.append(_non_negative_float(max(0.0, elapsed), "elapsed"))
    return BenchmarkResult(
        runs=count,
        minimum_seconds=min(durations),
        average_seconds=sum(durations) / count,
    )


def time_callable(
    function: Callable[..., R],
    *args: object,
    clock: Clock = time.perf_counter,
    **kwargs: object,
) -> tuple[R, float]:
    """Solution 1: return a result and its monotonic elapsed duration."""
    if not callable(function):
        raise TypeError("function must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    started = clock()
    result = function(*args, **kwargs)
    elapsed = _non_negative_float(max(0.0, clock() - started), "elapsed")
    return result, elapsed


def profile_call(
    function: Callable[..., R],
    *args: object,
    sort_by: str = "cumulative",
    lines: int = 8,
    **kwargs: object,
) -> str:
    """Profile one call and return a bounded, reviewable text report."""
    if not callable(function):
        raise TypeError("function must be callable")
    line_count = _positive_int(lines, "lines")

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        function(*args, **kwargs)
    finally:
        profiler.disable()

    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output)
    stats.strip_dirs().sort_stats(sort_by).print_stats(line_count)
    return output.getvalue()


def word_counts(text: str) -> dict[str, int]:
    """Return case-folded word frequencies for a deterministic workload."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    counts: Counter[str] = Counter()
    for match in _WORD.finditer(text):
        counts[match.group().casefold()] += 1
    return dict(counts)


def word_counts_split(text: str) -> dict[str, int]:
    """A simple alternative for whitespace-delimited text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    counts: Counter[str] = Counter()
    for word in text.split():
        counts[word.casefold()] += 1
    return dict(counts)


@lru_cache(maxsize=128)
def cached_fibonacci(n: int) -> int:
    """Solution 2: bounded memoization for a deterministic recurrence."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n cannot be negative")
    if n < 2:
        return n
    return cached_fibonacci(n - 1) + cached_fibonacci(n - 2)


def benchmark_many(
    named_functions: Mapping[str, Callable[[T], object]],
    argument: T,
    *,
    repeats: int = 5,
    clock: Clock = time.perf_counter,
) -> dict[str, BenchmarkResult]:
    """Solution 3: benchmark named callables in mapping order."""
    if not isinstance(named_functions, Mapping):
        raise TypeError("named_functions must be a mapping")
    result: dict[str, BenchmarkResult] = {}
    for name, function in named_functions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("function names must be non-empty strings")
        if not callable(function):
            raise TypeError("every mapped value must be callable")
        result[name] = benchmark(
            function,
            argument,
            repeats=repeats,
            clock=clock,
        )
    return result


def assert_within_budget(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    relative_budget: float = 0.10,
) -> None:
    """Raise ValueError when candidate is too much slower than baseline."""
    if not isinstance(baseline, BenchmarkResult) or not isinstance(
        candidate, BenchmarkResult
    ):
        raise TypeError("baseline and candidate must be BenchmarkResult values")
    budget = _non_negative_float(relative_budget, "relative_budget")
    allowed = baseline.average_seconds * (1.0 + budget)
    if candidate.average_seconds > allowed:
        raise ValueError(
            f"candidate exceeds budget: {candidate.average_seconds:.6f}s "
            f"> {allowed:.6f}s"
        )


def run_self_checks() -> None:
    """Check measurements, profiling, caching, and important edge cases."""
    clock_values = iter(
        value
        for pair in ((10.0, 10.25), (20.0, 20.5), (30.0, 30.125))
        for value in pair
    )
    measured = benchmark(
        lambda value: value * value,
        4,
        repeats=3,
        clock=lambda: next(clock_values),
    )
    assert measured == BenchmarkResult(3, 0.125, (0.25 + 0.5 + 0.125) / 3)

    result, elapsed = time_callable(
        lambda value: value + 1,
        4,
        clock=iter((1.0, 1.25)).__next__,
    )
    assert result == 5
    assert elapsed == 0.25

    assert word_counts("Python python's tools, Python!") == {
        "python": 2,
        "python's": 1,
        "tools": 1,
    }
    assert cached_fibonacci(0) == 0
    assert cached_fibonacci(10) == 55
    assert cached_fibonacci.cache_info().maxsize == 128

    values = iter(
        value
        for pair in ((1.0, 1.1), (2.0, 2.3), (3.0, 3.05), (4.0, 4.2))
        for value in pair
    )
    many = benchmark_many(
        {"first": lambda value: value, "second": lambda value: value + 1},
        10,
        repeats=2,
        clock=lambda: next(values),
    )
    assert tuple(many) == ("first", "second")
    assert math.isclose(many["first"].minimum_seconds, 0.1)
    assert math.isclose(many["first"].average_seconds, 0.2)
    assert math.isclose(many["second"].average_seconds, 0.125)

    profile = profile_call(word_counts, "one two two three", lines=3)
    assert "function calls" in profile
    assert "word_counts" in profile

    assert_within_budget(
        BenchmarkResult(2, 1.0, 1.1),
        BenchmarkResult(2, 1.05, 1.15),
        relative_budget=0.10,
    )
    for invalid_call in (
        lambda: BenchmarkResult(0, 0, 0),
        lambda: benchmark(None),  # type: ignore[arg-type]
        lambda: benchmark(lambda: None, repeats=0),
        lambda: benchmark(lambda: None, clock=None),  # type: ignore[arg-type]
        lambda: time_callable(None),  # type: ignore[arg-type]
        lambda: profile_call(lambda: None, lines=0),
        lambda: cached_fibonacci(True),  # type: ignore[arg-type]
        lambda: cached_fibonacci(-1),
        lambda: benchmark_many({"bad": None}, 1),  # type: ignore[arg-type]
        lambda: assert_within_budget(
            BenchmarkResult(1, 1, 1),
            BenchmarkResult(1, 2, 2),
            relative_budget=-0.1,
        ),
        lambda: assert_within_budget(
            BenchmarkResult(1, 1, 1),
            BenchmarkResult(1, 2, 2),
            relative_budget=0.1,
        ),
    ):
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, local demonstrations without relying on timing thresholds."""
    run_self_checks()

    workload = "Python performance " * 200
    print("Word count:", word_counts(workload))
    print("Cached Fibonacci(30):", cached_fibonacci(30))

    report = profile_call(word_counts, workload, lines=4)
    print("Profile excerpt:")
    print("\\n".join(report.splitlines()[:4]))

    # Real timings are informational only; they are not used as assertions.
    real = benchmark(word_counts_split, workload, repeats=3)
    print(
        "Benchmark:",
        f"runs={real.runs} min={real.minimum_seconds:.6f}s "
        f"avg={real.average_seconds:.6f}s",
    )
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
