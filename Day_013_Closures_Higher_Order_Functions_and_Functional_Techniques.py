"""Day 13: Closures, higher-order functions, and functional techniques.

Learning goals
--------------
1. Explain how a closure remembers values from an outer function.
2. Accept functions as arguments to separate "what to do" from "when to do it".
3. Return functions that carry validated configuration.
4. Compose small transformations into clear data-processing pipelines.
5. Use map, filter, comprehensions, and sorted with judgment.
6. Keep functional-style helpers pure unless their purpose is explicit state.

Teaching notes
--------------
- A higher-order function receives another function, returns a function, or
  both. sorted(items, key=...) is an everyday higher-order function.
- A closure is an inner function that remembers variables from the surrounding
  function after the outer function has finished running.
- Closures are useful for validated configuration: make_score_filter(80)
  returns a predicate that remembers the threshold.
- Functions are ordinary values. They can be stored in dictionaries, passed to
  other functions, and returned as results.
- Prefer a list comprehension for simple transformations and filters. Prefer a
  named helper or loop when validation, error handling, or several steps would
  make the expression hard to read.
- Pure functions are easier to test: same input, same output, no hidden output
  or mutation. Stateful closures are still useful, but make the state obvious.
- functools.partial can pre-fill arguments for an existing callable. A custom
  closure is often clearer when you also need validation or a readable name.
- Avoid clever chains that hide intent. Functional techniques should make data
  flow easier to reason about, not turn simple work into a puzzle.

Run this file with Python 3.10+ to execute deterministic examples and checks.
No third-party package is required.

Practice exercises
------------------
1. Write clamp_factory(minimum, maximum). It should return a function that
   clamps finite numbers into the inclusive range.
2. Write apply_discounts(prices, *discount_functions). Apply each discount
   function in order to every non-negative price and return rounded totals.
3. Write partition(items, predicate). Return a pair: items where predicate is
   True and items where it is False. Preserve order and support one-shot
   iterables.

Solutions appear below the main example.

Expert challenge: transformation pipeline mini-project
------------------------------------------------------
Build a reusable data-cleaning pipeline for imported student records. Each
step should be a pure function that receives and returns an immutable record.
Include validation, normalization, enrichment, filtering, and reporting. Add a
registry of named steps so a command-line layer can select a safe allowlist
without eval or arbitrary imports.

Solution guidance
-----------------
1. Model the external row separately from the validated domain record.
2. Keep pipeline steps small, typed, and individually testable.
3. Compose steps with a helper that stops at the first raised validation error.
4. Return accepted records and rejected rows with reasons instead of printing.
5. Test empty input, duplicate identifiers, whitespace normalization, invalid
   scores, unknown steps, and deterministic ordering.
6. Do not log private notes or student identifiers unless your policy allows it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from math import isfinite
from statistics import fmean
from typing import TypeVar

T = TypeVar("T")

NumberTransform = Callable[[float], float]
ScorePredicate = Callable[["ScoreRecord"], bool]
RecordTransform = Callable[["ScoreRecord"], "ScoreRecord"]


def _clean_text(value: str, field_name: str) -> str:
    """Return stripped, non-empty text with a useful error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _finite_number(value: int | float, field_name: str) -> float:
    """Normalize a finite numeric value while rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _score(value: int | float) -> float:
    """Return a finite score in the inclusive range 0 through 100."""
    normalized = _finite_number(value, "score")
    if not 0.0 <= normalized <= 100.0:
        raise ValueError("score must be between 0 and 100")
    return normalized


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """Immutable course progress record used by the functional examples."""

    student: str
    day: int
    topic: str
    score: float
    status: str = "recorded"

    def __post_init__(self) -> None:
        object.__setattr__(self, "student", _clean_text(self.student, "student"))
        if isinstance(self.day, bool) or not isinstance(self.day, int):
            raise TypeError("day must be an integer")
        if self.day <= 0:
            raise ValueError("day must be positive")
        object.__setattr__(self, "topic", _clean_text(self.topic, "topic"))
        object.__setattr__(self, "score", _score(self.score))
        object.__setattr__(self, "status", _clean_text(self.status, "status").casefold())


def make_multiplier(factor: int | float) -> NumberTransform:
    """Return a function that multiplies numbers by a remembered factor."""
    multiplier = _finite_number(factor, "factor")

    def multiply(value: int | float) -> float:
        return _finite_number(value, "value") * multiplier

    return multiply


def make_minimum_score_filter(minimum: int | float) -> ScorePredicate:
    """Return a predicate that keeps records at or above minimum."""
    threshold = _score(minimum)

    def has_enough_score(record: ScoreRecord) -> bool:
        if not isinstance(record, ScoreRecord):
            raise TypeError("record must be a ScoreRecord")
        return record.score >= threshold

    return has_enough_score


def make_status_step(passing_score: int | float) -> RecordTransform:
    """Return a pipeline step that labels records as passed or needs-review."""
    threshold = _score(passing_score)

    def add_status(record: ScoreRecord) -> ScoreRecord:
        if not isinstance(record, ScoreRecord):
            raise TypeError("record must be a ScoreRecord")
        status = "passed" if record.score >= threshold else "needs-review"
        return ScoreRecord(record.student, record.day, record.topic, record.score, status)

    return add_status


def compose(*functions: Callable[[object], object]) -> Callable[[object], object]:
    """Compose one-argument functions from left to right."""
    if not functions:
        raise ValueError("provide at least one function")
    if not all(callable(function) for function in functions):
        raise TypeError("every step must be callable")

    def composed(value: object) -> object:
        result = value
        for function in functions:
            result = function(result)
        return result

    return composed


def transform_records(
    records: Iterable[ScoreRecord],
    *steps: RecordTransform,
) -> tuple[ScoreRecord, ...]:
    """Apply each transformation step to each record in order."""
    if not steps:
        return tuple(records)
    if not all(callable(step) for step in steps):
        raise TypeError("every step must be callable")

    transformed: list[ScoreRecord] = []
    for record in records:
        if not isinstance(record, ScoreRecord):
            raise TypeError("records must contain ScoreRecord values")
        current = record
        for step in steps:
            current = step(current)
            if not isinstance(current, ScoreRecord):
                raise TypeError("record transform must return a ScoreRecord")
        transformed.append(current)
    return tuple(transformed)


def records_matching(
    records: Iterable[ScoreRecord],
    predicate: ScorePredicate,
) -> tuple[ScoreRecord, ...]:
    """Filter records with a named predicate and preserve original order."""
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    return tuple(record for record in records if predicate(record))


def summarize_by_student(records: Iterable[ScoreRecord]) -> dict[str, float]:
    """Group records by student and return average score per student."""
    grouped: dict[str, list[float]] = {}
    for record in records:
        if not isinstance(record, ScoreRecord):
            raise TypeError("records must contain ScoreRecord values")
        grouped.setdefault(record.student, []).append(record.score)
    return {
        student: round(fmean(scores), 2)
        for student, scores in sorted(grouped.items(), key=lambda item: item[0].casefold())
    }


def ranked_records(
    records: Iterable[ScoreRecord],
    *,
    key: Callable[[ScoreRecord], object] = lambda record: record.score,
    reverse: bool = True,
) -> tuple[ScoreRecord, ...]:
    """Return records sorted by a caller-supplied key."""
    if not callable(key):
        raise TypeError("key must be callable")
    return tuple(sorted(records, key=key, reverse=reverse))


def make_counter(start: int = 0) -> Callable[[], int]:
    """Return a small stateful closure for cases that need remembered state."""
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("start must be an integer")
    current = start

    def next_value() -> int:
        nonlocal current
        current += 1
        return current

    return next_value


# Practice exercise solutions


def clamp_factory(minimum: int | float, maximum: int | float) -> NumberTransform:
    """Solution 1: build a validated, reusable clamping function."""
    lower = _finite_number(minimum, "minimum")
    upper = _finite_number(maximum, "maximum")
    if lower > upper:
        raise ValueError("minimum cannot be greater than maximum")

    def clamp(value: int | float) -> float:
        number = _finite_number(value, "value")
        return min(max(number, lower), upper)

    return clamp


def make_percent_discount(percent: int | float) -> NumberTransform:
    """Return a function that reduces a price by a percentage."""
    rate = _finite_number(percent, "percent")
    if not 0.0 <= rate <= 100.0:
        raise ValueError("percent must be between 0 and 100")

    def discount(price: int | float) -> float:
        amount = _finite_number(price, "price")
        if amount < 0:
            raise ValueError("price cannot be negative")
        return amount * (1.0 - rate / 100.0)

    return discount


def add_fee(price: int | float, fee: int | float) -> float:
    """Add a non-negative fee to a non-negative price."""
    amount = _finite_number(price, "price")
    extra = _finite_number(fee, "fee")
    if amount < 0 or extra < 0:
        raise ValueError("price and fee cannot be negative")
    return amount + extra


def apply_discounts(
    prices: Iterable[int | float],
    *discount_functions: NumberTransform,
) -> tuple[float, ...]:
    """Solution 2: apply every pricing function to every price in order."""
    if not all(callable(function) for function in discount_functions):
        raise TypeError("every discount must be callable")

    totals: list[float] = []
    for price in prices:
        current = _finite_number(price, "price")
        if current < 0:
            raise ValueError("price cannot be negative")
        for discount in discount_functions:
            current = discount(current)
            if current < 0 or not isfinite(current):
                raise ValueError("discount must return a finite, non-negative price")
        totals.append(round(current, 2))
    return tuple(totals)


def partition(
    items: Iterable[object],
    predicate: Callable[[object], bool],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Solution 3: split an iterable into matching and non-matching items."""
    if not callable(predicate):
        raise TypeError("predicate must be callable")

    matched: list[object] = []
    unmatched: list[object] = []
    for item in items:
        if predicate(item):
            matched.append(item)
        else:
            unmatched.append(item)
    return tuple(matched), tuple(unmatched)


def run_self_checks() -> None:
    """Check closures, higher-order helpers, state, and edge cases."""
    double = make_multiplier(2)
    triple = make_multiplier(3)
    assert double(7) == 14
    assert triple(7) == 21

    records = (
        ScoreRecord(" Asha ", 13, "Closures", 92),
        ScoreRecord("Bikash", 13, "Functional techniques", 76),
        ScoreRecord("Mina", 12, "Typing", 84),
        ScoreRecord("Asha", 12, "Typing", 88),
    )
    add_status = make_status_step(80)
    labeled = transform_records(records, add_status)
    assert [record.status for record in labeled] == [
        "passed",
        "needs-review",
        "passed",
        "passed",
    ]

    high_scores = records_matching(records, make_minimum_score_filter(85))
    assert [record.student for record in high_scores] == ["Asha", "Asha"]
    assert summarize_by_student(records) == {
        "Asha": 90.0,
        "Bikash": 76.0,
        "Mina": 84.0,
    }
    assert [record.score for record in ranked_records(records)][:2] == [92.0, 88.0]
    assert [record.student for record in ranked_records(records, key=lambda item: item.student, reverse=False)] == [
        "Asha",
        "Asha",
        "Bikash",
        "Mina",
    ]

    clean_then_label = compose(
        lambda item: ScoreRecord(item.student.title(), item.day, item.topic, item.score),
        make_status_step(90),
    )
    assert clean_then_label(ScoreRecord("asha", 13, "closures", 91)).status == "passed"

    counter = make_counter(10)
    assert [counter(), counter(), counter()] == [11, 12, 13]

    clamp_score = clamp_factory(0, 100)
    assert [clamp_score(value) for value in (-5, 50, 130)] == [0.0, 50.0, 100.0]

    ten_percent = make_percent_discount(10)
    add_delivery = partial(add_fee, fee=2.5)
    assert apply_discounts([100, 50], ten_percent, add_delivery) == (92.5, 47.5)

    even, odd = partition((number for number in range(5)), lambda item: item % 2 == 0)
    assert even == (0, 2, 4)
    assert odd == (1, 3)

    invalid_calls = (
        lambda: ScoreRecord("", 13, "Topic", 80),
        lambda: ScoreRecord("Asha", True, "Topic", 80),  # type: ignore[arg-type]
        lambda: ScoreRecord("Asha", 13, "Topic", 101),
        lambda: make_multiplier(float("nan")),
        lambda: make_minimum_score_filter(-1),
        lambda: compose(),
        lambda: compose(lambda value: value, "bad"),  # type: ignore[arg-type]
        lambda: transform_records([object()], add_status),  # type: ignore[list-item]
        lambda: records_matching(records, None),  # type: ignore[arg-type]
        lambda: ranked_records(records, key=None),  # type: ignore[arg-type]
        lambda: make_counter(True),  # type: ignore[arg-type]
        lambda: clamp_factory(10, 5),
        lambda: clamp_factory(0, float("inf")),
        lambda: make_percent_discount(101),
        lambda: apply_discounts([10, -1], ten_percent),
        lambda: apply_discounts([10], lambda price: float("nan")),
        lambda: partition([1, 2], None),  # type: ignore[arg-type]
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without external services."""
    run_self_checks()

    records = (
        ScoreRecord("Asha", 13, "Closures", 92),
        ScoreRecord("Bikash", 13, "Closures", 76),
        ScoreRecord("Mina", 13, "Closures", 84),
    )
    labeled = transform_records(records, make_status_step(80))
    strong_results = records_matching(labeled, make_minimum_score_filter(85))

    print("Labeled records:", labeled)
    print("Strong results:", strong_results)
    print("Student averages:", summarize_by_student(labeled))
    print("Discounted prices:", apply_discounts([120, 75], make_percent_discount(15)))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
