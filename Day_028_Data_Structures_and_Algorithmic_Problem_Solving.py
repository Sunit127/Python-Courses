"""Day 28: Data structures and algorithmic problem solving.

Learning goals
--------------
1. Match a problem to a list, set, mapping, deque, heap, or graph representation.
2. Use invariants to make stack, queue, and heap operations predictable.
3. Analyze time and space trade-offs with useful Big-O estimates.
4. Implement binary search and breadth-first search without off-by-one errors.
5. Turn a larger problem into small, testable helpers with clear contracts.

Teaching notes
--------------
A data structure is a promise about how data is stored and what operations are
cheap. Lists give indexed access; sets and dictionaries give average O(1)
membership or lookup; deque supports O(1) operations at both ends; heapq gives
the smallest priority item in O(log n) insertion/removal. These are practical
averages, not guarantees for every adversarial input.

An algorithm's complexity describes how work grows with input size. Binary search
cuts a sorted search space in half on every step (O(log n)); a full scan is O(n).
Breadth-first search visits each reachable node and edge once (O(V + E)) when
adjacency lists are used. State the preconditions—such as sorted input—so
callers cannot accidentally use a fast algorithm incorrectly.

Prefer a simple representation and measure before optimizing. An explicit
invariant makes correctness review easier: in a queue, items leave in insertion
order; in binary search, the answer is never outside the remaining bounds; in
BFS, a node is enqueued at most once.

Runnable examples
-----------------
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Iterable, Mapping, TypeVar


T = TypeVar("T")


def _positive_int(value: int, *, field: str) -> int:
    """Return a positive integer or raise a useful boundary error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def binary_search(sorted_values: list[T], target: T) -> int | None:
    """Return the first index of target in sorted_values, or None.

    Preconditions: sorted_values is sorted in ascending order. The algorithm
    keeps the invariant that target, if present, remains in [left, right).
    """
    left, right = 0, len(sorted_values)
    while left < right:
        middle = left + (right - left) // 2
        value = sorted_values[middle]
        if value < target:
            left = middle + 1
        elif value > target:
            right = middle
        else:
            # Continue left so duplicates return their first index.
            right = middle
    return left if left < len(sorted_values) and sorted_values[left] == target else None


def first_repeated(values: Iterable[T]) -> T | None:
    """Return the first value whose second occurrence is encountered."""
    seen: set[T] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


@dataclass
class WorkQueue:
    """A FIFO queue with an explicit empty-queue contract."""

    _items: deque[str]

    @classmethod
    def from_items(cls, items: Iterable[str] = ()) -> "WorkQueue":
        queue = cls(deque())
        for item in items:
            queue.put(item)
        return queue

    def put(self, item: str) -> None:
        cleaned = item.strip()
        if not cleaned:
            raise ValueError("queue items must not be blank")
        self._items.append(cleaned)

    def get(self) -> str | None:
        return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class PrioritizedJob:
    priority: int
    name: str

    def __post_init__(self) -> None:
        _positive_int(self.priority, field="priority")
        if not self.name.strip():
            raise ValueError("job name must not be blank")


def run_jobs_by_priority(jobs: Iterable[PrioritizedJob]) -> tuple[str, ...]:
    """Process the lowest numeric priority first, breaking ties by name."""
    heap: list[tuple[int, str]] = []
    for job in jobs:
        heapq.heappush(heap, (job.priority, job.name.strip()))
    return tuple(heapq.heappop(heap)[1] for _ in range(len(heap)))


def shortest_hops(
    graph: Mapping[str, Iterable[str]], start: str, goal: str
) -> tuple[str, ...] | None:
    """Find a shortest unweighted path with breadth-first search.

    Missing vertices are treated as having no outgoing edges. Neighbors are
    visited in sorted order to make the result deterministic for demonstrations.
    """
    if start == goal:
        return (start,)
    queue: deque[str] = deque([start])
    previous: dict[str, str | None] = {start: None}

    while queue:
        current = queue.popleft()
        for neighbor in sorted(set(graph.get(current, ()))):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            if neighbor == goal:
                path: list[str] = [goal]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])  # type: ignore[arg-type]
                return tuple(reversed(path))
            queue.append(neighbor)
    return None


def top_k_frequent(values: Iterable[T], k: int) -> tuple[T, ...]:
    """Return up to k values with highest frequency, then first-seen order.

    Values are ranked by count descending and first appearance ascending.
    """
    _positive_int(k, field="k")
    counts: dict[T, int] = {}
    first_seen: dict[T, int] = {}
    for index, value in enumerate(values):
        counts[value] = counts.get(value, 0) + 1
        first_seen.setdefault(value, index)
    ranked = sorted(counts, key=lambda value: (-counts[value], first_seen[value]))
    return tuple(ranked[:k])


# Practice exercises
# ------------------
# 1. Implement a stack with append/pop and return None when it is empty.
# 2. Adapt binary_search to return the insertion index for a missing target.
# 3. Add a "has_cycle(graph)" helper using DFS and a three-state visitation map.
# 4. Compare list.pop(0) with deque.popleft() on a large workload and explain
#    the measured difference rather than guessing from a single timing.

# Solutions / clear guidance
# --------------------------
# 1. Use list[str] as the storage; append pushes and pop() removes the latest
#    item. Check "if not items" before popping to keep the empty contract.
# 2. Keep the same [left, right) invariant. When the loop ends, left is the
#    first valid insertion position, whether or not the target is present.
# 3. DFS marks a node "visiting" before descending and "done" after returning.
#    Seeing a "visiting" neighbor means the current path contains a cycle.
# 4. list.pop(0) shifts every remaining element (O(n)); deque.popleft() removes
#    from the left in O(1). Benchmark after choosing a realistic input size.

def stack_demo(items: Iterable[str]) -> tuple[str, ...]:
    """Solution for exercise 1: return items in last-in, first-out order."""
    stack = [item for item in items]
    popped: list[str] = []
    while stack:
        popped.append(stack.pop())
    return tuple(popped)


def insertion_index(sorted_values: list[T], target: T) -> int:
    """Solution for exercise 2: return target's leftmost insertion position."""
    left, right = 0, len(sorted_values)
    while left < right:
        middle = left + (right - left) // 2
        if sorted_values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


def has_cycle(graph: Mapping[str, Iterable[str]]) -> bool:
    """Solution for exercise 3: detect a directed cycle with DFS."""
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in done:
            return False
        visiting.add(node)
        for neighbor in graph.get(node, ()):
            if visit(neighbor):
                return True
        visiting.remove(node)
        done.add(node)
        return False

    return any(visit(node) for node in graph if node not in done)


def run_self_checks() -> None:
    assert binary_search([1, 2, 2, 4], 2) == 1
    assert binary_search([], 10) is None
    assert first_repeated(["a", "b", "a"]) == "a"
    assert first_repeated(["a", "b"]) is None

    queue = WorkQueue.from_items([" first "])
    queue.put("second")
    assert len(queue) == 2
    assert queue.get() == "first"
    assert queue.get() == "second"
    assert queue.get() is None

    jobs = [PrioritizedJob(2, "compile"), PrioritizedJob(1, "test"), PrioritizedJob(1, "lint")]
    assert run_jobs_by_priority(jobs) == ("lint", "test", "compile")
    assert shortest_hops({"A": ["B", "C"], "B": ["D"], "C": ["D"]}, "A", "D") == (
        "A",
        "B",
        "D",
    )
    assert shortest_hops({"A": []}, "A", "Z") is None
    assert top_k_frequent(["x", "y", "x", "z", "y", "x"], 2) == ("x", "y")
    assert stack_demo(["a", "b", "c"]) == ("c", "b", "a")
    assert insertion_index([1, 3, 3, 8], 3) == 1
    assert insertion_index([1, 3, 3, 8], 4) == 3
    assert has_cycle({"A": ["B"], "B": ["A"]})
    assert not has_cycle({"A": ["B"], "B": []})

    for invalid in (0, -1, True):
        try:
            _positive_int(invalid, field="value")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid positive integer was accepted")


def main() -> None:
    run_self_checks()
    print("Day 28 checks passed.")
    print(
        "Priority order:",
        run_jobs_by_priority(
            [
                PrioritizedJob(2, "compile"),
                PrioritizedJob(1, "test"),
                PrioritizedJob(1, "lint"),
            ]
        ),
    )
    print(
        "Shortest path:",
        shortest_hops(
            {"home": ["park", "shop"], "park": ["office"], "shop": ["office"]},
            "home",
            "office",
        ),
    )


if __name__ == "__main__":
    main()
