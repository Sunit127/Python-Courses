"""Day 14: Iterators, generators, and lazy data pipelines.

Learning goals
--------------
1. Explain the iterator protocol: iter() creates an iterator and next() advances it.
2. Write generators that produce values on demand with yield.
3. Preserve memory by composing lazy transformations instead of building lists.
4. Handle one-shot iterators, exhaustion, sentinels, and invalid pipeline settings.
5. Separate a streaming input layer from a pure aggregation or reporting core.

Teaching notes
--------------
- An iterable can create an iterator; an iterator remembers its current position.
  A list is reusable, but an iterator is usually one-shot.
- next(iterator, default) is useful at a boundary where exhaustion is expected.
  A normal next(iterator) raises StopIteration when no value remains.
- A generator function pauses at each yield and resumes with its local state
  intact. It does not do work until a caller asks for the next value.
- Lazy pipelines are valuable for large files, sockets, and database cursors:
  each item can flow through several stages without storing the whole stream.
- Use a list when you need random access or repeated traversal. Use an iterator
  when you can consume values once and want bounded memory.
- A generator should not silently swallow malformed data. Choose a documented
  policy: reject the stream, skip known bad records, or yield an error object.
- itertools provides well-tested building blocks such as islice, chain, and
  accumulate. Compose them with small named functions for readable pipelines.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Write take(iterable, count) to lazily yield at most count values. Reject a
   negative count.
2. Write unique_everseen(iterable) to yield each hashable value once, keeping
   first-seen order.
3. Write chunked(iterable, size) to yield tuples of up to size values. The final
   chunk may be smaller than size.

Solutions appear below the examples.

Expert challenge: streaming sensor report
------------------------------------------
Extend summarize_readings so it can read an unbounded source and expose a
second generator that yields only readings above a threshold. Keep parsing,
filtering, and aggregation separate. Add a policy for malformed lines that
reports line numbers without materializing all valid readings. A production
version could read from a file or socket and emit periodic summaries.
"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from itertools import accumulate, chain, islice
from math import isfinite
from typing import TypeVar

T = TypeVar("T")


def countdown(start: int) -> Iterator[int]:
    """Yield integers from start down to one without building a list."""
    if start < 0:
        raise ValueError("start must be non-negative")
    while start:
        yield start
        start -= 1


def take(iterable: Iterable[T], count: int) -> Iterator[T]:
    """Solution 1: lazily yield at most count values."""
    if count < 0:
        raise ValueError("count cannot be negative")
    yield from islice(iterable, count)


def unique_everseen(iterable: Iterable[T]) -> Iterator[T]:
    """Solution 2: yield each hashable value once in first-seen order."""
    seen: set[T] = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item


def chunked(iterable: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """Solution 3: yield bounded tuples, including a short final chunk."""
    if size <= 0:
        raise ValueError("size must be positive")

    iterator = iter(iterable)
    while True:
        chunk = tuple(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def windowed(iterable: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """Yield each complete sliding window of a one-shot iterable."""
    if size <= 0:
        raise ValueError("size must be positive")

    iterator = iter(iterable)
    window = deque(islice(iterator, size), maxlen=size)
    if len(window) < size:
        return

    yield tuple(window)
    for item in iterator:
        window.append(item)
        yield tuple(window)


def pipeline(
    source: Iterable[T],
    *stages: Callable[[Iterable[object]], Iterable[object]],
) -> Iterable[object]:
    """Compose lazy stages without forcing any stage into a list."""
    current: Iterable[object] = source
    for stage in stages:
        current = stage(current)
    return current


def parse_reading(line: str) -> tuple[str, float]:
    """Parse a sensor line in the form sensor-name, value."""
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 2 or not fields[0]:
        raise ValueError("expected sensor-name, value")

    try:
        value = float(fields[1])
    except ValueError as error:
        raise ValueError("reading value must be numeric") from error

    if not isfinite(value):
        raise ValueError("reading value must be finite")
    return fields[0], value


def valid_readings(
    lines: Iterable[str],
    rejected_lines: list[int] | None = None,
) -> Iterator[tuple[str, float]]:
    """Lazily parse lines, optionally recording known malformed line numbers."""
    for line_number, line in enumerate(lines, start=1):
        try:
            yield parse_reading(line)
        except ValueError:
            if rejected_lines is None:
                raise
            rejected_lines.append(line_number)


def readings_above(
    readings: Iterable[tuple[str, float]],
    threshold: float,
) -> Iterator[tuple[str, float]]:
    """Yield readings above threshold without buffering the source."""
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")
    for sensor, value in readings:
        if value > threshold:
            yield sensor, value


def summarize_readings(lines: Iterable[str]) -> dict[str, object]:
    """Aggregate a stream in one pass while preserving rejected line numbers."""
    rejected_lines: list[int] = []
    count = 0
    total = 0.0
    smallest: float | None = None
    largest: float | None = None

    for _, value in valid_readings(lines, rejected_lines):
        count += 1
        total += value
        smallest = value if smallest is None else min(smallest, value)
        largest = value if largest is None else max(largest, value)

    return {
        "count": count,
        "total": round(total, 2),
        "average": round(total / count, 2) if count else None,
        "smallest": smallest,
        "largest": largest,
        "rejected_lines": rejected_lines,
    }


def run_self_checks() -> None:
    """Check laziness, normal output, and important boundary cases."""
    countdown_values = countdown(3)
    assert iter(countdown_values) is countdown_values
    assert next(countdown_values) == 3
    assert list(countdown_values) == [2, 1]
    assert list(countdown_values) == []

    assert list(take(range(10), 3)) == [0, 1, 2]
    assert list(take(range(3), 0)) == []
    assert list(unique_everseen(["a", "b", "a", "c", "b"])) == [
        "a",
        "b",
        "c",
    ]
    assert list(chunked(range(5), 2)) == [(0, 1), (2, 3), (4,)]
    assert list(windowed([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]
    assert list(windowed([1, 2], 3)) == []

    pulls = {"count": 0}

    def source() -> Iterator[int]:
        for value in range(4):
            pulls["count"] += 1
            yield value

    lazy_result = pipeline(
        source(),
        lambda values: (value + 1 for value in values),
        lambda values: (value * 10 for value in values if value % 2 == 0),
    )
    assert pulls["count"] == 0
    assert list(lazy_result) == [20, 40]
    assert pulls["count"] == 4

    running_totals = accumulate([1, 2, 3])
    assert list(running_totals) == [1, 3, 6]
    assert list(chain(["a"], ["b", "c"])) == ["a", "b", "c"]

    rejected: list[int] = []
    parsed = list(valid_readings(["a, 2.5", "broken", "b, 3"], rejected))
    assert parsed == [("a", 2.5), ("b", 3.0)]
    assert rejected == [2]
    assert list(readings_above(parsed, 2.5)) == [("b", 3.0)]

    report = summarize_readings(
        ["a, 2.5", "broken", "b, 3", "a, 4.5", "c, nan"]
    )
    assert report == {
        "count": 3,
        "total": 10.0,
        "average": 3.33,
        "smallest": 2.5,
        "largest": 4.5,
        "rejected_lines": [2, 5],
    }

    invalid_calls = (
        lambda: list(countdown(-1)),
        lambda: list(take([], -1)),
        lambda: list(chunked([], 0)),
        lambda: list(windowed([], 0)),
        lambda: list(readings_above([], float("inf"))),
        lambda: parse_reading("sensor, nan"),
        lambda: list(valid_readings(["bad"])),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input must raise ValueError")


def main() -> None:
    """Run safe, deterministic demonstrations without user input."""
    run_self_checks()

    print("Countdown:", list(countdown(5)))
    print("Chunks:", list(chunked(range(1, 8), 3)))
    print(
        "Pipeline:",
        list(
            pipeline(
                range(1, 7),
                lambda values: (value * value for value in values),
                lambda values: (value for value in values if value > 10),
            )
        ),
    )
    print("Running totals:", list(accumulate([4, 6, 3])))
    print("Combined labels:", list(chain(["sensor-a"], ["sensor-b", "sensor-c"])))

    sample_lines = [
        "sensor-a, 21.5",
        "sensor-b, 18",
        "not-a-reading",
        "sensor-a, 24.25",
    ]
    report = summarize_readings(sample_lines)
    print("Sensor report:", report)
    print(
        "Above 20:",
        list(readings_above(valid_readings(sample_lines, []), 20)),
    )
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
