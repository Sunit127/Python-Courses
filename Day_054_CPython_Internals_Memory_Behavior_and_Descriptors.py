"""Day 54: CPython internals, memory behavior, and descriptors.

This lesson connects Python syntax to CPython's object model: names reference
objects, containers own references, reference counting usually reclaims acyclic
objects promptly, and the cyclic collector handles unreachable cycles. It also
builds descriptors, the attribute-lookup protocol behind properties and many
framework APIs.

The examples are standard-library only and intentionally small. CPython facts
are useful diagnostics, not portable language guarantees. Code that must run on
PyPy or another interpreter should test behavior rather than a bytecode opcode
or an exact byte count.

Learning goals
--------------
1. Separate identity, equality, shallow size, and object ownership.
2. Explain trade-offs among __slots__, interning, weak references, and cycles.
3. Implement a validating data descriptor and a lazy non-data descriptor.
4. Inspect bytecode and allocations without treating diagnostics as contracts.
5. Label implementation-specific observations and provide portable fallbacks.

Teaching notes
--------------
- Assignment binds a name; it does not copy the object. id(value) is an
  identity token during that object's lifetime, == asks for value equality, and
  sys.getsizeof reports only one object's shallow footprint.
- CPython decrements a reference count when a reference disappears. A cycle can
  keep every member reachable from another member, so the cyclic GC periodically
  finds unreachable generations. Never make correctness depend on collection
  timing or a particular reference-count number.
- __slots__ can remove an instance __dict__ and save overhead for many
  fixed-shape objects, but it also removes arbitrary attributes unless a
  __dict__ slot is explicitly included.
- sys.intern can share equal identifier strings. Intern bounded, reused values;
  interning untrusted input can retain a large pool.
- A descriptor defining __set__ or __delete__ is a data descriptor and wins over
  an instance dictionary. A descriptor defining only __get__ is non-data and
  can be shadowed by an instance attribute. This is how a property enforces an
  invariant while a cached property stores a computed value.
- weak references do not keep a target alive. WeakValueDictionary is suitable
  for optional caches, never for durable state; callers must handle a miss.
- dis and tracemalloc are diagnostic tools. Opcode names, allocation totals,
  and object sizes vary by Python version, build, allocator, and workload.

Run with Python 3.10+. Self-checks cover descriptor precedence, slots,
weak-cache lifetime, interning, cycle collection, bytecode inspection, and a
bounded allocation trace.
"""

from __future__ import annotations

import dis
import gc
import sys
import tracemalloc
import weakref
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """A shallow, interpreter-aware observation of one object."""

    type_name: str
    object_id: int
    shallow_size: int
    gc_tracked: bool


def snapshot(value: object) -> MemorySnapshot:
    """Observe one object without claiming to measure its whole graph."""
    return MemorySnapshot(
        type_name=type(value).__name__,
        object_id=id(value),
        shallow_size=sys.getsizeof(value),
        gc_tracked=gc.is_tracked(value),
    )


def same_interned_text(left: str, right: str) -> bool:
    """Compare two texts after explicitly placing them in the intern pool."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("left and right must be strings")
    left_interned = sys.intern(left)
    right_interned = sys.intern(right)
    return left_interned is right_interned


class PositiveInt:
    """A data descriptor that stores a positive integer in a private slot."""

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> int | PositiveInt:
        if instance is None:
            return self
        try:
            return getattr(instance, self.storage_name)
        except AttributeError as error:
            raise AttributeError(f"{self.public_name} has not been set") from error

    def __set__(self, instance: object, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{self.public_name} must be an integer")
        if value <= 0:
            raise ValueError(f"{self.public_name} must be positive")
        object.__setattr__(instance, self.storage_name, value)


class Account:
    """An object whose balance invariant is enforced by PositiveInt."""

    balance = PositiveInt()

    def __init__(self, opening_balance: int) -> None:
        self.balance = opening_balance


class CachedProperty(Generic[T]):
    """A non-data descriptor that computes once and caches in __dict__."""

    def __init__(self, function: Callable[[object], T]) -> None:
        self.function = function
        self.public_name = function.__name__
        self.storage_name = f"_{self.public_name}_cached"

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.public_name = name
        self.storage_name = f"_{name}_cached"

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> T | CachedProperty[T]:
        if instance is None:
            return self
        values = vars(instance)
        if self.storage_name not in values:
            values[self.storage_name] = self.function(instance)
        return values[self.storage_name]


class Report:
    """A lazy report showing non-data descriptor shadowing safely."""

    def __init__(self, values: list[int]) -> None:
        self.values = list(values)
        self.compute_count = 0

    @CachedProperty
    def total(self) -> int:
        self.compute_count += 1
        return sum(self.values)


class SlottedPoint:
    """A fixed-shape instance without an instance dictionary."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def distance_squared(self) -> float:
        return self.x * self.x + self.y * self.y


@dataclass
class WeakEntry:
    """A weak-referenceable value for a cache demonstration."""

    key: str
    payload: tuple[int, ...]


def weak_cache_lifecycle() -> tuple[int, int]:
    """Return cache sizes before and after the only strong value is deleted."""
    cache: weakref.WeakValueDictionary[str, WeakEntry] = weakref.WeakValueDictionary()
    entry = WeakEntry("sample", (1, 2, 3))
    cache[entry.key] = entry
    before = len(cache)
    del entry
    gc.collect()
    return before, len(cache)


def make_unreachable_cycle() -> int:
    """Create a list cycle and return the collector's reported count."""
    first: list[object] = []
    second: list[object] = [first]
    first.append(second)
    del first, second
    return gc.collect()


def bytecode_names(function: Callable[..., object]) -> tuple[str, ...]:
    """Return opcode names for diagnostics, not for compatibility assertions."""
    if not callable(function):
        raise TypeError("function must be callable")
    return tuple(item.opname for item in dis.get_instructions(function))


def allocation_delta(factory: Callable[[], object], count: int = 100) -> int:
    """Measure a bounded allocation difference for one repeatable factory."""
    if not callable(factory):
        raise TypeError("factory must be callable")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count <= 0:
        raise ValueError("count must be positive")

    tracemalloc.start()
    try:
        before = tracemalloc.take_snapshot()
        values = [factory() for _ in range(count)]
        after = tracemalloc.take_snapshot()
        difference = sum(
            stat.size_diff for stat in after.compare_to(before, "filename")
        )
        del values
        return difference
    finally:
        tracemalloc.stop()


def descriptor_demo() -> dict[str, object]:
    """Return a compact, observable descriptor demonstration."""
    account = Account(10)
    account.balance = 25
    report = Report([2, 3, 5])
    first_total = report.total
    second_total = report.total
    return {
        "balance": account.balance,
        "total": first_total,
        "same_cached_value": first_total == second_total,
        "compute_count": report.compute_count,
        "slot_distance_squared": SlottedPoint(3, 4).distance_squared(),
    }


def run_self_checks() -> None:
    """Exercise invariants and edge cases without relying on timing."""
    account = Account(3)
    account.balance = 8
    assert account.balance == 8
    for invalid in (0, True):
        try:
            account.balance = invalid  # type: ignore[assignment]
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid balances must be rejected")

    report = Report([1, 2, 3])
    assert report.total == 6 and report.total == 6
    assert report.compute_count == 1
    assert report.__dict__["_total_cached"] == 6

    point = SlottedPoint(3, 4)
    assert point.distance_squared() == 25.0
    try:
        point.label = "origin"
    except AttributeError:
        pass
    else:
        raise AssertionError("__slots__ must reject undeclared attributes")

    assert same_interned_text(
        "course_identifier", "".join(["course", "_identifier"])
    )
    before, after = weak_cache_lifecycle()
    assert (before, after) == (1, 0)
    assert make_unreachable_cycle() >= 0

    opcodes = bytecode_names(lambda value: value + 1)
    assert opcodes
    assert any(name in opcodes for name in ("RETURN_VALUE", "RETURN_CONST"))
    assert allocation_delta(lambda: SlottedPoint(1, 2), 20) >= 0

    observed = snapshot(point)
    assert observed.type_name == "SlottedPoint"
    assert observed.shallow_size > 0
    assert isinstance(observed.object_id, int)


def demo() -> None:
    """Print a deterministic tour of the lesson's ideas."""
    print("Descriptor demo:", descriptor_demo())
    print("Interned identity:", same_interned_text("status", "status"))
    print("Bytecode:", bytecode_names(lambda value: value + 1))
    print("Cycle collection found:", make_unreachable_cycle())
    print("Slotted snapshot:", snapshot(SlottedPoint(3, 4)))


# Practice exercises
# ------------------
# 1. Extend PositiveInt into RangeInt with inclusive lower and upper bounds.
#    Reject bool, preserve the public name from __set_name__, and include the
#    bounds in error messages.
# 2. Add "__dict__" to SlottedPoint.__slots__. Which new behavior is possible,
#    and how does sys.getsizeof change on your Python build? Record the version.
# 3. Make Report.total invalidating: changing values must delete the cached key.
#    Decide whether direct assignment to total is rejected or supported.
# 4. Use tracemalloc to compare 10,000 tuples with 10,000 small dataclass
#    instances. Explain why shallow size is not a full object-graph profile.
# 5. Build a bounded synthetic graph and identify unreachable cycles. Explain
#    why ownership boundaries, weak back-links, and explicit close methods are
#    usually preferable to forcing gc.collect in application code.

# Solutions / clear guidance
# --------------------------
# 1. Store the private name in __set_name__; check
#    isinstance(value, int) and not isinstance(value, bool), then validate
#    lower <= value <= upper. Use TypeError for type failures and ValueError
#    for range failures.
# 2. A "__dict__" slot permits point.label = ... but gives each instance a
#    dictionary. Compare snapshots from the same interpreter; allocator and
#    build flags make absolute byte counts non-portable.
# 3. Let a values property setter delete the cached key. A data descriptor for
#    total can reject writes; a non-data descriptor can intentionally allow an
#    override. Test empty values and repeated reads.
# 4. Keep both collections alive through the second snapshot, compare
#    size_diff by filename or traceback, and report methodology plus the Python
#    version instead of claiming one universal number.
# 5. Use gc.get_objects only for a bounded teaching graph. In production,
#    define ownership and close external resources explicitly; weak references
#    are for optional back-links and caches, not durability.

# Expert challenge / mini-project
# --------------------------------
# Build a local memory-inspection harness:
# * Define GraphNode with an explicit owner, an optional weak parent, and a
#   bounded list of uniquely named children.
# * Add descriptor-based configuration fields with validation, defaults, and a
#   read-only computed property. Include __set_name__ and useful errors.
# * Report shallow sizes, gc tracking, type counts, and top tracemalloc
#   differences before and after a deterministic synthetic workload. Do not
#   expose raw addresses as stable identifiers.
# * Repeat create/use/discard cycles, collect only at a test boundary, and fail
#   when retained strong references exceed an explicit budget.
# * Run on CPython and, if available, PyPy. Label interpreter-specific facts
#   and keep behavioral tests portable.
#
# A strong solution separates measurement from policy, documents allocator and
# interpreter assumptions, and keeps output small enough for code review. The
# goal is to reason about lifetime, lookup, and evidence-based optimization,
# not to memorize one release's opcode table.

if __name__ == "__main__":
    run_self_checks()
    demo()
