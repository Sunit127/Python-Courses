"""Day 8: Object-oriented design with classes and dataclasses.

Learning goals
--------------
1. Model related state and behavior with small, focused classes.
2. Distinguish instance attributes, class attributes, methods, and properties.
3. Use dataclasses to remove boilerplate from data-centered objects.
4. Protect invariants with validation and controlled mutation.
5. Prefer composition when one object coordinates other objects.

Teaching notes
--------------
- A class defines a new type; each instance owns its instance attributes.
  Methods receive the current instance as self and can enforce rules whenever
  state changes.
- A class attribute is shared by the class. Do not use a shared mutable class
  attribute for per-instance data.
- Dataclasses generate methods such as __init__, __repr__, and __eq__ from type-
  annotated fields. They do not make validation automatic; use __post_init__.
- Use field(default_factory=list) or another factory for mutable defaults so
  each instance receives its own collection.
- A property exposes method-backed access with attribute syntax. It is useful
  for computed values or when reads and writes must preserve an invariant.
- A leading underscore communicates that an attribute is an implementation
  detail. Python relies on cooperation rather than absolute privacy.
- Composition means that an object contains or coordinates other objects. A
  TaskBoard has Task objects, which keeps responsibilities clearer than a deep
  inheritance hierarchy.
- frozen=True is useful for value objects that should not change. slots=True
  prevents accidental new attributes and can reduce per-instance memory.
- Keep input/output outside domain classes when possible. Domain methods should
  receive Python values, return useful results, and raise specific exceptions.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Create a Rectangle dataclass with positive width and height, plus area and
   perimeter properties.
2. Create a Student class that validates scores from 0 through 100 and exposes
   scores as an immutable snapshot and average as a read-only property.
3. Write next_pending(board) to return the first incomplete task or None.

Solutions appear below the main example.

Expert challenge: dependency-aware project planner
--------------------------------------------------
Extend Task with a unique identifier and a tuple of dependency identifiers.
A task may start only when all its dependencies are complete. Reject missing
dependencies, self-dependencies, and dependency cycles. Keep graph validation
in a ProjectPlan coordinator rather than in Task. Add tests for a diamond-
shaped dependency graph, a missing identifier, a cycle, and a fully completed
project.

Solution guidance:
1. Store tasks in a dictionary keyed by identifier.
2. Validate references when adding a task.
3. Detect cycles with depth-first search and two sets: visiting and visited.
4. Compute ready tasks by checking every dependency's completed state.
5. Return tuples or copies from public APIs so callers cannot bypass rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import ClassVar


def _clean_nonempty(text: str, field_name: str) -> str:
    """Return stripped text or raise a useful validation error."""
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


@dataclass(frozen=True, slots=True)
class CourseLabel:
    """An immutable value object suitable for dictionary keys and sets."""

    name: str
    level: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_nonempty(self.name, "name"))
        object.__setattr__(
            self,
            "level",
            _clean_nonempty(self.level, "level").casefold(),
        )


@dataclass(slots=True)
class Task:
    """A validated unit of work.

    tags uses a factory rather than a shared list. The tuple stored after
    validation prevents callers from changing tags behind this object's back.
    """

    title: str
    estimated_minutes: int = 25
    tags: tuple[str, ...] = field(default_factory=tuple)
    completed: bool = False

    def __post_init__(self) -> None:
        self.title = _clean_nonempty(self.title, "title")
        if isinstance(self.estimated_minutes, bool) or not isinstance(
            self.estimated_minutes, int
        ):
            raise TypeError("estimated_minutes must be an integer")
        if self.estimated_minutes <= 0:
            raise ValueError("estimated_minutes must be positive")

        normalized_tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in self.tags:
            cleaned = _clean_nonempty(raw_tag, "tag").casefold()
            if cleaned not in seen:
                seen.add(cleaned)
                normalized_tags.append(cleaned)
        self.tags = tuple(normalized_tags)

        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a Boolean")

    def mark_complete(self) -> None:
        """Complete the task. Repeating the operation is harmless."""
        self.completed = True

    def reopen(self) -> None:
        """Move the task back to pending state."""
        self.completed = False

    def has_tag(self, tag: str) -> bool:
        """Return whether a normalized tag belongs to this task."""
        return _clean_nonempty(tag, "tag").casefold() in self.tags


class TaskBoard:
    """Coordinate Task objects and enforce board-wide rules."""

    boards_created: ClassVar[int] = 0

    def __init__(self, name: str) -> None:
        self._name = _clean_nonempty(name, "name")
        self._tasks: dict[str, Task] = {}
        type(self).boards_created += 1

    @property
    def name(self) -> str:
        """Return the validated board name."""
        return self._name

    @property
    def tasks(self) -> tuple[Task, ...]:
        """Return an immutable snapshot of tasks in insertion order."""
        return tuple(self._tasks.values())

    @property
    def progress_percent(self) -> float:
        """Return completion percentage; an empty board is 0% complete."""
        if not self._tasks:
            return 0.0
        completed_count = sum(task.completed for task in self._tasks.values())
        return round(completed_count / len(self._tasks) * 100, 1)

    @classmethod
    def from_titles(cls, name: str, titles: list[str]) -> TaskBoard:
        """Construct a board containing default tasks for the supplied titles."""
        board = cls(name)
        for title in titles:
            board.add(Task(title))
        return board

    def add(self, task: Task) -> None:
        """Add a task while requiring case-insensitively unique titles."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task instance")
        key = task.title.casefold()
        if key in self._tasks:
            raise ValueError(f"task title already exists: {task.title!r}")
        self._tasks[key] = task

    def create_task(
        self,
        title: str,
        *,
        estimated_minutes: int = 25,
        tags: tuple[str, ...] = (),
    ) -> Task:
        """Create, add, and return a Task through one convenient boundary."""
        task = Task(title, estimated_minutes, tags)
        self.add(task)
        return task

    def get(self, title: str) -> Task:
        """Return a task by title or raise LookupError."""
        key = _clean_nonempty(title, "title").casefold()
        try:
            return self._tasks[key]
        except KeyError as error:
            raise LookupError(f"unknown task: {title!r}") from error

    def complete(self, title: str) -> None:
        """Complete a task selected through the board."""
        self.get(title).mark_complete()

    def pending(self) -> tuple[Task, ...]:
        """Return pending tasks without exposing the internal dictionary."""
        return tuple(task for task in self._tasks.values() if not task.completed)

    def find_by_tag(self, tag: str) -> tuple[Task, ...]:
        """Return all tasks containing tag."""
        cleaned = _clean_nonempty(tag, "tag")
        return tuple(task for task in self._tasks.values() if task.has_tag(cleaned))

    def remaining_minutes(self) -> int:
        """Return the total estimate for incomplete tasks."""
        return sum(task.estimated_minutes for task in self.pending())

    def summary(self) -> dict[str, object]:
        """Return presentation-neutral board data."""
        return {
            "name": self.name,
            "task_count": len(self._tasks),
            "completed_count": len(self._tasks) - len(self.pending()),
            "progress_percent": self.progress_percent,
            "remaining_minutes": self.remaining_minutes(),
        }


# Practice exercise solutions


@dataclass(slots=True)
class Rectangle:
    """Solution 1: a validated data-centered class."""

    width: float
    height: float

    def __post_init__(self) -> None:
        self.width = float(self.width)
        self.height = float(self.height)
        if not math.isfinite(self.width) or not math.isfinite(self.height):
            raise ValueError("width and height must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class Student:
    """Solution 2: protect a mutable score collection behind methods."""

    def __init__(self, name: str) -> None:
        self._name = _clean_nonempty(name, "name")
        self._scores: list[float] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    @property
    def average(self) -> float | None:
        if not self._scores:
            return None
        return round(sum(self._scores) / len(self._scores), 2)

    def add_score(self, score: float) -> None:
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("score must be finite")
        if not 0 <= numeric_score <= 100:
            raise ValueError("score must be between 0 and 100")
        self._scores.append(numeric_score)


def next_pending(board: TaskBoard) -> Task | None:
    """Solution 3: return the first pending task, if one exists."""
    pending = board.pending()
    return pending[0] if pending else None


def run_self_checks() -> None:
    """Check object behavior, isolation, errors, and important edge cases."""
    python_label = CourseLabel(" Python ", " BEGINNER ")
    assert python_label == CourseLabel("Python", "beginner")
    assert hash(python_label) == hash(CourseLabel("Python", "beginner"))

    first = Task(" Write tests ", 30, (" Python ", "testing", "python"))
    second = Task("Document API", tags=("docs",))
    assert first.title == "Write tests"
    assert first.tags == ("python", "testing")
    assert not first.completed

    board = TaskBoard(" Course release ")
    board.add(first)
    board.add(second)
    assert board.name == "Course release"
    assert board.tasks == (first, second)
    assert board.progress_percent == 0.0
    assert board.remaining_minutes() == 55
    assert board.find_by_tag(" PYTHON ") == (first,)
    assert next_pending(board) is first

    board.complete("write TESTS")
    assert first.completed
    assert board.progress_percent == 50.0
    assert board.remaining_minutes() == 25
    first.reopen()
    assert board.progress_percent == 0.0

    quick = TaskBoard.from_titles("Quick start", ["Read", "Practice"])
    assert [task.title for task in quick.tasks] == ["Read", "Practice"]

    rectangle = Rectangle(3, 4)
    assert rectangle.area == 12
    assert rectangle.perimeter == 14

    student = Student("Asha")
    assert student.average is None
    student.add_score(80)
    student.add_score(90)
    assert student.scores == (80.0, 90.0)
    assert student.average == 85.0

    unrelated_a = Task("A")
    unrelated_b = Task("B")
    assert unrelated_a.tags is not unrelated_b.tags or unrelated_a.tags == ()

    invalid_calls = (
        lambda: Task(""),
        lambda: Task("Bad estimate", 0),
        lambda: Task("Boolean estimate", True),
        lambda: Task("Bad tag", tags=("",)),
        lambda: Task("Bad completion", completed=1),
        lambda: board.add(Task("write tests")),
        lambda: board.get("missing"),
        lambda: board.add("not a task"),  # type: ignore[arg-type]
        lambda: Rectangle(-1, 2),
        lambda: Rectangle(float("nan"), 2),
        lambda: student.add_score(101),
        lambda: student.add_score(float("nan")),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (LookupError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without user input."""
    run_self_checks()

    board = TaskBoard("Day 8 practice")
    board.create_task(
        "Read object-oriented design notes",
        estimated_minutes=20,
        tags=("python", "reading"),
    )
    board.create_task(
        "Implement the exercises",
        estimated_minutes=45,
        tags=("python", "practice"),
    )
    board.complete("Read object-oriented design notes")

    print("Board:", board.summary())
    print("Pending:", [task.title for task in board.pending()])
    print("Python tasks:", [task.title for task in board.find_by_tag("python")])
    print("Next task:", next_pending(board).title if next_pending(board) else None)
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
