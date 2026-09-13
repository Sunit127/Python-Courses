"""Day 12: Type hints, generics, and static analysis.

Learning goals
--------------
1. Annotate functions, collections, callables, and optional values precisely.
2. Model constrained data with Literal, TypedDict, and type aliases.
3. Preserve relationships between input and output types with TypeVar.
4. Build reusable generic classes without giving up type information.
5. Narrow unknown values safely with TypeGuard.
6. Understand what static analysis can find and what runtime checks must enforce.

Teaching notes
--------------
- Type hints describe an interface for readers, editors, and static type
  checkers. Python normally does not enforce annotations at runtime, so
  validation is still required at untrusted boundaries.
- Prefer precise built-in generics such as list[str], dict[str, int], and
  tuple[int, ...]. Use Iterable[T] when a function only needs to loop, and a
  concrete collection only when its behavior is required.
- A union such as str | None means either value is valid. Narrow the union with
  an explicit check before using string-only operations.
- object means an unknown value that must be narrowed before use. Any disables
  most useful checking and should be limited to truly dynamic boundaries.
- Literal describes a small set of accepted values. TypedDict describes the
  expected keys and value types of dictionary-shaped data. Neither performs
  runtime validation by itself.
- TypeVar connects types. A function receiving Iterable[T] and returning
  T | None promises that its result, when present, has the same type as an
  input item.
- Generic classes keep type information across stored values. KeyedStore[K, V]
  remembers both its key type and its value type.
- overload documents return types that depend on literal arguments. The final
  implementation still needs one runtime body covering every overload.
- TypeGuard lets a validation function tell a checker that an object has a more
  specific type after the function returns True. The guard must be truthful.
- Run a checker such as pyright or mypy in addition to executing tests. A type
  checker finds incompatible calls without running them; tests find behavioral
  errors and validate runtime boundaries.
- Use "from __future__ import annotations" to defer annotation evaluation. This
  supports forward references and keeps annotations from doing unnecessary
  runtime work.

Run this file with Python 3.10+ to execute deterministic examples and checks.
No third-party package is required.

Practice exercises
------------------
1. Implement generic chunked(items, size). Return immutable chunks, preserve
   order, support one-shot iterables, and reject Boolean or non-positive sizes.
2. Implement index_unique(items, key). Preserve each item's precise type and
   raise ValueError when two items produce the same key.
3. Implement valid_lesson_payloads(items). Keep only dictionaries accepted by
   is_lesson_payload and return a tuple whose narrowed type is LessonPayload.

Solutions appear below the main example.

Expert challenge: typed configuration loader
--------------------------------------------
Build a configuration loader that accepts unknown JSON-shaped input and
produces immutable application settings. Use TypedDict for the external shape,
dataclasses for validated domain values, Literal for environments, and a
generic Result[T] or specific exceptions for failures. Support separate
development, test, and production settings without using Any.

Solution guidance
-----------------
1. Treat decoded JSON as object at the boundary and narrow every nested value.
2. Separate shape validation from domain validation and normalization.
3. Report a path such as "database.port" with each validation error.
4. Never store secrets in repr output, logs, or exception messages.
5. Add overloads only when callers receive meaningfully different return types.
6. Test missing and extra keys, Booleans passed as integers, invalid ports,
   empty strings, unknown environments, and a fully valid configuration.
7. Run both the test suite and a strict static checker in continuous integration.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from typing import Generic, Literal, TypeGuard, TypedDict, TypeVar, overload


LessonStatus = Literal["passed", "needs-review"]
T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LessonPayload(TypedDict):
    """Dictionary shape accepted at an external data boundary."""

    day: int
    title: str
    score: int | float
    status: LessonStatus


def _clean_text(value: str, field_name: str) -> str:
    """Return stripped, non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _positive_int(value: int, field_name: str) -> int:
    """Reject Booleans even though bool is a subclass of int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _score(value: int | float) -> float:
    """Normalize a finite score in the inclusive range 0 through 100."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("score must be a number")
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 100.0:
        raise ValueError("score must be finite and between 0 and 100")
    return normalized


@dataclass(frozen=True, slots=True)
class LessonResult:
    """Validated domain value; annotations alone would not enforce its rules."""

    day: int
    title: str
    score: float
    status: LessonStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "day", _positive_int(self.day, "day"))
        object.__setattr__(self, "title", _clean_text(self.title, "title"))
        object.__setattr__(self, "score", _score(self.score))
        if self.status not in ("passed", "needs-review"):
            raise ValueError("status must be 'passed' or 'needs-review'")


@dataclass(frozen=True, slots=True)
class ProgressSummary:
    """Typed aggregate returned by summarize_results."""

    lessons: int
    passed: int
    average_score: float | None


def summarize_results(results: Iterable[LessonResult]) -> ProgressSummary:
    """Summarize any iterable, including a generator consumed exactly once."""
    total = 0
    passed = 0
    score_total = 0.0

    for result in results:
        if not isinstance(result, LessonResult):
            raise TypeError("results must contain LessonResult values")
        total += 1
        passed += result.status == "passed"
        score_total += result.score

    average = score_total / total if total else None
    return ProgressSummary(total, passed, average)


@overload
def select_result(
    results: Iterable[LessonResult],
    predicate: Callable[[LessonResult], bool],
    *,
    required: Literal[True],
) -> LessonResult:
    ...


@overload
def select_result(
    results: Iterable[LessonResult],
    predicate: Callable[[LessonResult], bool],
    *,
    required: Literal[False] = False,
) -> LessonResult | None:
    ...


def select_result(
    results: Iterable[LessonResult],
    predicate: Callable[[LessonResult], bool],
    *,
    required: bool = False,
) -> LessonResult | None:
    """Return the first match, optionally raising when no value matches."""
    for result in results:
        if not isinstance(result, LessonResult):
            raise TypeError("results must contain LessonResult values")
        if predicate(result):
            return result
    if required:
        raise LookupError("no lesson result matched")
    return None


def is_lesson_payload(value: object) -> TypeGuard[LessonPayload]:
    """Narrow unknown input only after checking its complete runtime shape."""
    if not isinstance(value, dict):
        return False
    if set(value) != {"day", "title", "score", "status"}:
        return False

    day = value.get("day")
    title = value.get("title")
    score = value.get("score")
    status = value.get("status")
    return (
        isinstance(day, int)
        and not isinstance(day, bool)
        and day > 0
        and isinstance(title, str)
        and bool(title.strip())
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and isfinite(float(score))
        and 0.0 <= float(score) <= 100.0
        and status in ("passed", "needs-review")
    )


def lesson_from_payload(value: object) -> LessonResult:
    """Convert unknown dictionary-shaped input into a validated domain value."""
    if not is_lesson_payload(value):
        raise TypeError("value is not a valid lesson payload")
    return LessonResult(
        day=value["day"],
        title=value["title"],
        score=float(value["score"]),
        status=value["status"],
    )


class KeyedStore(Generic[K, V]):
    """Generic in-memory store that retains its key and value types."""

    def __init__(self, key: Callable[[V], K]) -> None:
        if not callable(key):
            raise TypeError("key must be callable")
        self._key = key
        self._values: dict[K, V] = {}

    def add(self, value: V) -> K:
        """Store a value once and return its derived key."""
        key = self._key(value)
        if key in self._values:
            raise ValueError(f"duplicate key: {key!r}")
        self._values[key] = value
        return key

    def get(self, key: K) -> V | None:
        """Return the value for key, or None when it is absent."""
        return self._values.get(key)

    def snapshot(self) -> dict[K, V]:
        """Return a defensive copy so callers cannot mutate the store."""
        return dict(self._values)

    def __iter__(self) -> Iterator[V]:
        return iter(self._values.values())

    def __len__(self) -> int:
        return len(self._values)


# Practice exercise solutions


def chunked(items: Iterable[T], size: int) -> tuple[tuple[T, ...], ...]:
    """Solution 1: group any iterable into immutable chunks."""
    size = _positive_int(size, "size")
    chunks: list[tuple[T, ...]] = []
    current: list[T] = []

    for item in items:
        current.append(item)
        if len(current) == size:
            chunks.append(tuple(current))
            current = []

    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def index_unique(
    items: Iterable[V],
    key: Callable[[V], K],
) -> dict[K, V]:
    """Solution 2: build an index without silently replacing duplicates."""
    if not callable(key):
        raise TypeError("key must be callable")

    index: dict[K, V] = {}
    for item in items:
        item_key = key(item)
        if item_key in index:
            raise ValueError(f"duplicate key: {item_key!r}")
        index[item_key] = item
    return index


def valid_lesson_payloads(
    items: Iterable[object],
) -> tuple[LessonPayload, ...]:
    """Solution 3: filter unknown values and preserve the narrowed type."""
    return tuple(item for item in items if is_lesson_payload(item))


def run_self_checks() -> None:
    """Check typing examples, runtime boundaries, and important edge cases."""
    first = LessonResult(11, " Protocols ", 92, "passed")
    second = LessonResult(12, "Typing", 78, "needs-review")

    summary = summarize_results(item for item in (first, second))
    assert summary == ProgressSummary(2, 1, 85.0)
    assert summarize_results([]) == ProgressSummary(0, 0, None)

    assert select_result([first, second], lambda item: item.day == 12) == second
    assert select_result([first], lambda item: item.day == 99) is None
    try:
        select_result([first], lambda item: False, required=True)
    except LookupError as error:
        assert str(error) == "no lesson result matched"
    else:
        raise AssertionError("required selection must raise LookupError")

    payload: object = {
        "day": 12,
        "title": " Type hints ",
        "score": 88,
        "status": "passed",
    }
    assert is_lesson_payload(payload)
    assert lesson_from_payload(payload) == LessonResult(
        12, "Type hints", 88.0, "passed"
    )
    assert not is_lesson_payload({"day": True, "title": "Bad", "score": 80, "status": "passed"})
    assert not is_lesson_payload({"day": 1, "title": "", "score": 80, "status": "passed"})
    assert not is_lesson_payload({"day": 1, "title": "Bad", "score": 101, "status": "passed"})
    assert not is_lesson_payload({"day": 1, "title": "Bad", "score": 80, "status": "unknown"})
    assert not is_lesson_payload({"day": 1, "title": "Bad", "score": 80, "status": "passed", "extra": 1})

    store: KeyedStore[int, LessonResult] = KeyedStore(lambda item: item.day)
    assert store.add(first) == 11
    assert store.add(second) == 12
    assert len(store) == 2
    assert store.get(12) == second
    assert store.get(99) is None
    snapshot = store.snapshot()
    snapshot.clear()
    assert len(store) == 2
    assert tuple(store) == (first, second)

    assert chunked(range(5), 2) == ((0, 1), (2, 3), (4,))
    assert chunked((str(number) for number in range(3)), 5) == (("0", "1", "2"),)
    assert chunked([], 3) == ()

    by_title = index_unique([first, second], lambda item: item.title.casefold())
    assert by_title["typing"] == second

    payloads = valid_lesson_payloads(
        [payload, None, {"day": 0}, "not a dictionary"]
    )
    assert len(payloads) == 1
    assert payloads[0]["day"] == 12

    invalid_calls = (
        lambda: LessonResult(True, "Bad", 80, "passed"),  # type: ignore[arg-type]
        lambda: LessonResult(1, "", 80, "passed"),
        lambda: LessonResult(1, "Bad", float("nan"), "passed"),
        lambda: LessonResult(1, "Bad", 80, "unknown"),  # type: ignore[arg-type]
        lambda: summarize_results([object()]),  # type: ignore[list-item]
        lambda: lesson_from_payload({"day": 1}),
        lambda: chunked([1, 2], 0),
        lambda: chunked([1, 2], True),
        lambda: index_unique(["A", "a"], str.casefold),
        lambda: KeyedStore(None),  # type: ignore[arg-type]
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")

    duplicate_store: KeyedStore[int, LessonResult] = KeyedStore(
        lambda item: item.day
    )
    duplicate_store.add(first)
    try:
        duplicate_store.add(LessonResult(11, "Duplicate", 100, "passed"))
    except ValueError as error:
        assert "duplicate key" in str(error)
    else:
        raise AssertionError("duplicate store key must raise ValueError")


def main() -> None:
    """Run safe, deterministic demonstrations without external services."""
    run_self_checks()

    lessons = [
        LessonResult(11, "Protocols", 92, "passed"),
        LessonResult(12, "Type hints", 88, "passed"),
    ]
    store: KeyedStore[int, LessonResult] = KeyedStore(lambda item: item.day)
    for lesson in lessons:
        store.add(lesson)

    summary = summarize_results(store)
    print("Stored days:", tuple(store.snapshot()))
    print("Average score:", summary.average_score)
    print("Chunks:", chunked((lesson.title for lesson in store), 1))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
