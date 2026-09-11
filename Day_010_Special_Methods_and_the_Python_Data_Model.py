"""Day 10: Special methods and the Python data model.

Learning goals
--------------
1. Connect Python syntax such as len(), iteration, indexing, and operators to
   the special methods that implement it.
2. Build value objects with predictable representations, equality, ordering,
   and hashing.
3. Implement container and iterator protocols without surprising callers.
4. Return NotImplemented for unsupported binary operations.
5. Choose a small, coherent protocol instead of adding every possible dunder.

Teaching notes
--------------
- Python syntax is protocol-driven. len(value) asks value.__len__(), value[index]
  asks value.__getitem__(index), and left + right tries the operands' addition
  methods. Prefer the public syntax; call dunder methods directly only when
  implementing or studying a protocol.
- repr(value) should be unambiguous and useful while debugging. str(value)
  should be readable for users. Never include secrets in either representation.
- Equality and hashing must agree: objects that compare equal must have the
  same hash. Mutable objects should normally be unhashable because changing a
  hashed value can corrupt a set or dictionary lookup.
- A binary special method should return NotImplemented when it does not support
  the other operand's type. Python can then try the reflected operation or
  raise an accurate TypeError. NotImplemented is a value, not an exception.
- A container is iterable when __iter__ returns an iterator. An iterator also
  implements __next__ and returns itself from __iter__. Once exhausted, it must
  continue raising StopIteration.
- Sequence-like objects should make negative indexes and slices behave like
  built-in sequences. Storing data in a tuple makes defensive snapshots easy.
- Special methods define promises users already understand from built-in types.
  Implement only behavior with a clear meaning, and preserve those expectations.
- Dataclasses can generate several data-model methods, but generated behavior
  is still an API decision. Validate fields and consider frozen value objects.
- __slots__ can prevent accidental attributes and save memory for many small
  instances. It is an implementation choice, not a security boundary.

Run this file with Python 3.10+ to execute deterministic examples and checks.

Practice exercises
------------------
1. Implement Version(major, minor, patch) as an immutable, orderable value
   object. Reject Booleans and negative or non-integer components.
2. Implement Countdown(start) as a stateful iterator yielding start through 1.
   It must remain exhausted after raising StopIteration.
3. Implement TagSet(values), an immutable, case-insensitive container with
   len(), membership, iteration in sorted order, and a useful repr().

Solutions appear below the main examples.

Expert challenge: immutable polynomial mini-project
---------------------------------------------------
Build a Polynomial whose coefficients are stored from constant term upward.
Normalize redundant trailing zeros while preserving the zero polynomial.
Implement repr(), str(), equality, hashing, iteration, len(), indexing,
evaluation through __call__, addition, and multiplication. Support adding a
number by returning NotImplemented for unrelated types, and make reflected
addition work. Add tests for normalization, the zero polynomial, negative
indexes, slices, mismatched operand types, and algebraic identities.

Solution guidance
-----------------
1. Convert coefficients to finite floats or Decimals at construction and store
   an immutable tuple.
2. Remove trailing zeros while len(coefficients) > 1.
3. Let tuple indexing handle negative indexes and slices consistently.
4. For addition, pad the shorter coefficient tuple with zeros.
5. For multiplication, allocate len(a) + len(b) - 1 zeros and accumulate every
   pairwise coefficient product.
6. Evaluate with Horner's method by visiting coefficients in reverse order.
7. Base equality and hashing on the normalized coefficient tuple.
8. Keep formatting separate from the arithmetic so repr and str stay testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import total_ordering
import math
from typing import overload


@total_ordering
class Money:
    """Immutable-style monetary value with explicit currency rules."""

    __slots__ = ("_amount", "_currency")

    def __init__(self, amount: Decimal | int | str, currency: str) -> None:
        if isinstance(amount, bool) or not isinstance(
            amount, (Decimal, int, str)
        ):
            raise TypeError("amount must be a Decimal, integer, or numeric text")
        try:
            parsed_amount = Decimal(str(amount))
        except InvalidOperation as error:
            raise ValueError("amount must be valid numeric text") from error
        if not parsed_amount.is_finite():
            raise ValueError("amount must be finite")

        if not isinstance(currency, str):
            raise TypeError("currency must be a string")
        normalized_currency = currency.strip().upper()
        if len(normalized_currency) != 3 or not normalized_currency.isalpha():
            raise ValueError("currency must contain exactly three letters")

        self._amount = parsed_amount
        self._currency = normalized_currency

    @property
    def amount(self) -> Decimal:
        return self._amount

    @property
    def currency(self) -> str:
        return self._currency

    def __repr__(self) -> str:
        return f"Money({str(self.amount)!r}, {self.currency!r})"

    def __str__(self) -> str:
        return f"{self.amount:.2f} {self.currency}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return (self.amount, self.currency) == (other.amount, other.currency)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount < other.amount

    def __hash__(self) -> int:
        return hash((self.amount, self.currency))

    def __bool__(self) -> bool:
        return self.amount != 0

    def __add__(self, other: object) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: object) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: object) -> Money:
        if isinstance(factor, bool) or not isinstance(factor, (Decimal, int)):
            return NotImplemented
        decimal_factor = Decimal(factor)
        if not decimal_factor.is_finite():
            raise ValueError("factor must be finite")
        return Money(self.amount * decimal_factor, self.currency)

    def __rmul__(self, factor: object) -> Money:
        return self * factor

    def _require_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"currency mismatch: {self.currency} and {other.currency}"
            )


@dataclass(frozen=True, slots=True)
class Reading:
    """One immutable item stored by ReadingLog."""

    title: str
    pages: int

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title cannot be empty")
        if isinstance(self.pages, bool) or not isinstance(self.pages, int):
            raise TypeError("pages must be an integer")
        if self.pages <= 0:
            raise ValueError("pages must be positive")
        object.__setattr__(self, "title", self.title.strip())


class ReadingLog(Sequence[Reading]):
    """A read-only sequence with familiar indexing and slicing behavior."""

    __slots__ = ("_readings",)

    def __init__(self, readings: Iterable[Reading] = ()) -> None:
        materialized = tuple(readings)
        if not all(isinstance(reading, Reading) for reading in materialized):
            raise TypeError("every item must be a Reading")
        self._readings = materialized

    def __len__(self) -> int:
        return len(self._readings)

    @overload
    def __getitem__(self, index: int) -> Reading:
        ...

    @overload
    def __getitem__(self, index: slice) -> ReadingLog:
        ...

    def __getitem__(self, index: int | slice) -> Reading | ReadingLog:
        selected = self._readings[index]
        if isinstance(index, slice):
            return ReadingLog(selected)
        return selected

    def __iter__(self) -> Iterator[Reading]:
        return iter(self._readings)

    def __reversed__(self) -> Iterator[Reading]:
        return reversed(self._readings)

    def __repr__(self) -> str:
        return f"ReadingLog({list(self._readings)!r})"

    @property
    def total_pages(self) -> int:
        return sum(reading.pages for reading in self)


@dataclass(frozen=True, slots=True)
class Vector2D:
    """Small numeric value object demonstrating operator protocols."""

    x: float
    y: float

    def __post_init__(self) -> None:
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("coordinates must be finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)

    def __bool__(self) -> bool:
        return self.x != 0 or self.y != 0

    def __add__(self, other: object) -> Vector2D:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: object) -> Vector2D:
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return NotImplemented
        numeric_scalar = float(scalar)
        if not math.isfinite(numeric_scalar):
            raise ValueError("scalar must be finite")
        return Vector2D(self.x * numeric_scalar, self.y * numeric_scalar)

    def __rmul__(self, scalar: object) -> Vector2D:
        return self * scalar

    def __matmul__(self, other: object) -> float:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return self.x * other.x + self.y * other.y


# Practice exercise solutions


@total_ordering
@dataclass(frozen=True, slots=True)
class Version:
    """Solution 1: an immutable semantic version core."""

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (
            self.major,
            self.minor,
            self.patch,
        ) < (
            other.major,
            other.minor,
            other.patch,
        )


class Countdown(Iterator[int]):
    """Solution 2: a stateful, single-use iterator."""

    __slots__ = ("_current",)

    def __init__(self, start: int) -> None:
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("start must be an integer")
        if start < 0:
            raise ValueError("start cannot be negative")
        self._current = start

    def __iter__(self) -> Countdown:
        return self

    def __next__(self) -> int:
        if self._current == 0:
            raise StopIteration
        value = self._current
        self._current -= 1
        return value


class TagSet:
    """Solution 3: immutable, case-insensitive container."""

    __slots__ = ("_tags",)

    def __init__(self, values: Iterable[str] = ()) -> None:
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise TypeError("every tag must be a string")
            cleaned = value.strip().casefold()
            if not cleaned:
                raise ValueError("tags cannot be empty")
            normalized.add(cleaned)
        self._tags = frozenset(normalized)

    def __len__(self) -> int:
        return len(self._tags)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._tags))

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and value.strip().casefold() in self._tags

    def __repr__(self) -> str:
        return f"TagSet({list(self)!r})"


def run_self_checks() -> None:
    """Check protocol behavior and representative edge cases."""
    price = Money("12.50", "usd")
    shipping = Money("7.50", "USD")
    assert repr(price) == "Money('12.50', 'USD')"
    assert str(price) == "12.50 USD"
    assert price + shipping == Money("20.00", "USD")
    assert price - shipping == Money("5.00", "USD")
    assert 2 * price == Money("25.00", "USD")
    assert price < Money("13", "USD")
    assert bool(price)
    assert not Money("0", "USD")
    assert len({price, Money("12.5", "usd")}) == 1

    log = ReadingLog(
        [
            Reading("Fluent Python", 1014),
            Reading("Python Cookbook", 706),
            Reading("Architecture Patterns", 304),
        ]
    )
    assert len(log) == 3
    assert log[-1].title == "Architecture Patterns"
    assert [reading.title for reading in reversed(log)][0] == (
        "Architecture Patterns"
    )
    assert log[:2].total_pages == 1720
    assert Reading("Python Cookbook", 706) in log

    vector = Vector2D(3, 4)
    assert tuple(vector) == (3.0, 4.0)
    assert abs(vector) == 5
    assert vector + Vector2D(1, -1) == Vector2D(4, 3)
    assert vector * 2 == Vector2D(6, 8)
    assert 2 * vector == Vector2D(6, 8)
    assert vector @ Vector2D(2, 1) == 10
    assert not Vector2D(0, 0)

    versions = [Version(1, 10, 0), Version(1, 2, 5), Version(2, 0, 0)]
    assert [str(version) for version in sorted(versions)] == [
        "1.2.5",
        "1.10.0",
        "2.0.0",
    ]

    countdown = Countdown(3)
    assert list(countdown) == [3, 2, 1]
    assert list(countdown) == []

    tags = TagSet([" Python ", "TESTING", "python"])
    assert len(tags) == 2
    assert list(tags) == ["python", "testing"]
    assert "PyThOn" in tags
    assert 42 not in tags
    assert repr(tags) == "TagSet(['python', 'testing'])"

    invalid_calls = (
        lambda: Money(1.5, "USD"),
        lambda: Money("nan", "USD"),
        lambda: Money("1", "US"),
        lambda: price + Money("1", "EUR"),
        lambda: price * 1.5,
        lambda: Reading("", 10),
        lambda: Reading("Book", True),
        lambda: ReadingLog([object()]),
        lambda: Vector2D(float("inf"), 0),
        lambda: vector + (1, 2),
        lambda: Version(True, 0, 0),
        lambda: Version(1, -1, 0),
        lambda: Countdown(-1),
        lambda: TagSet([""]),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, self-contained demonstrations."""
    run_self_checks()

    subtotal = Money("18.50", "USD")
    print("Invoice:", subtotal + Money("2.50", "USD"))

    log = ReadingLog(
        [
            Reading("Python Data Model Notes", 24),
            Reading("Protocol Practice", 16),
        ]
    )
    print("Reading log:", log)
    print("Total pages:", log.total_pages)
    print("First title:", log[0].title)

    direction = Vector2D(3, 4)
    print("Vector coordinates:", tuple(direction))
    print("Vector magnitude:", abs(direction))
    print("Scaled vector:", 0 * direction)

    print("Countdown:", list(Countdown(3)))
    print("Tags:", list(TagSet(["Data Model", "Python", "python"])))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
