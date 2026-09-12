"""Day 11: Protocols, duck typing, and interface design.

Learning goals
--------------
1. Explain how duck typing lets callers depend on behavior instead of ancestry.
2. Describe structural interfaces with typing.Protocol.
3. Keep protocols narrow so implementations remain easy to replace and test.
4. Adapt existing callables and objects to a protocol without changing them.
5. Understand the limits of runtime-checkable protocols.

Teaching notes
--------------
- Duck typing asks whether an object supports the operation a caller needs:
  "if it behaves like a sink here, use it as a sink." Inheriting from a shared
  base class is optional.
- A Protocol gives that structural expectation a name for type checkers. A
  class satisfies it by having compatible members; explicit inheritance is not
  required.
- Accept the smallest useful interface. A formatter needs format(), while a
  sink needs write(). Combining unrelated operations creates rigid designs.
- Protocols document collaboration; abstract base classes can also provide
  shared implementation and runtime enforcement. Choose an ABC when controlled
  construction or shared behavior matters, and a Protocol for loose coupling.
- @runtime_checkable permits isinstance checks, but those checks only inspect
  whether named attributes exist. They do not validate method signatures,
  return types, invariants, or behavior. Tests and static analysis still matter.
- Standard collection interfaces such as Iterable, Sequence, and Mapping are
  protocols in everyday use. Accept an Iterable when the function only loops;
  do not require a list without a list-specific need.
- Adapters translate an existing interface into the one a caller expects.
  PrefixFormatter wraps another formatter; CallbackSink turns a function into
  an EventSink.
- Dependency injection means collaborators arrive as arguments instead of being
  constructed deep inside business logic. Tests can then use small in-memory
  fakes with no network, file, or database side effects.
- Structural compatibility is not semantic compatibility. A write() method
  that deletes data still has the right shape but the wrong promise. Document
  effects and failure behavior as part of the interface contract.

Run this file with Python 3.10+ to execute deterministic examples and checks.

Practice exercises
------------------
1. Define a Named protocol and sorted_names(items). Accept unrelated objects
   with a string name, return normalized names sorted case-insensitively, and
   reject objects that do not satisfy the protocol.
2. Implement PrefixFormatter, which adds a validated prefix to the result of
   any EventFormatter without changing the wrapped formatter.
3. Implement CallbackSink, an adapter that forwards each record to a callable.

Solutions appear below the main example.

Expert challenge: pluggable audit pipeline
------------------------------------------
Build an audit pipeline with narrow protocols for EventSource, EventFilter,
EventFormatter, and EventSink. Sources yield events, filters may reject them,
formatters serialize accepted events, and one or more sinks store the records.
Support an in-memory implementation of every boundary so tests require no
external services.

Solution guidance
-----------------
1. Keep Event immutable and free of input/output behavior.
2. Materialize one-shot iterables only when the pipeline must revisit them.
3. Define ordering and partial-failure policy when one of several sinks fails.
4. Add an adapter for line-oriented files, but inject an already-open text
   stream so ownership and cleanup remain explicit.
5. Test an empty source, filter rejection, formatter failure, sink failure,
   deterministic ordering, and a retry that must not duplicate stored records.
6. Document whether records may contain private data; redact before formatting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
from typing import Protocol, runtime_checkable


def _clean_text(value: str, field_name: str) -> str:
    """Return stripped, non-empty text with a useful error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event data shared across formatters and sinks."""

    kind: str
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _clean_text(self.kind, "kind").casefold(),
        )
        object.__setattr__(self, "message", _clean_text(self.message, "message"))

        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in self.details:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each detail must be a (key, value) tuple")
            raw_key, raw_value = item
            key = _clean_text(raw_key, "detail key").casefold()
            value = _clean_text(raw_value, "detail value")
            if key in seen:
                raise ValueError(f"duplicate detail key: {key!r}")
            seen.add(key)
            normalized.append((key, value))

        object.__setattr__(self, "details", tuple(normalized))


@runtime_checkable
class EventFormatter(Protocol):
    """Structural interface for converting one event to text."""

    def format(self, event: Event, /) -> str:
        """Return one complete record without performing output."""


@runtime_checkable
class EventSink(Protocol):
    """Structural interface for storing or forwarding one text record."""

    def write(self, record: str, /) -> None:
        """Accept one complete record or raise an appropriate exception."""


class LineFormatter:
    """Readable formatter that satisfies EventFormatter implicitly."""

    def format(self, event: Event, /) -> str:
        suffix = ""
        if event.details:
            pairs = ", ".join(f"{key}={value}" for key, value in event.details)
            suffix = f" ({pairs})"
        return f"{event.kind}: {event.message}{suffix}"


class JsonFormatter:
    """Deterministic JSON formatter with no output side effects."""

    def format(self, event: Event, /) -> str:
        return json.dumps(
            {
                "details": dict(event.details),
                "kind": event.kind,
                "message": event.message,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class ListSink:
    """In-memory sink; it uses no EventSink inheritance."""

    def __init__(self) -> None:
        self._records: list[str] = []

    @property
    def records(self) -> tuple[str, ...]:
        """Return an immutable snapshot of stored records."""
        return tuple(self._records)

    def write(self, record: str, /) -> None:
        self._records.append(_clean_text(record, "record"))


class EventPublisher:
    """Format events once and publish each record to every configured sink."""

    def __init__(
        self,
        formatter: EventFormatter,
        sinks: Iterable[EventSink],
    ) -> None:
        if not isinstance(formatter, EventFormatter):
            raise TypeError("formatter must provide format(event)")

        materialized_sinks = tuple(sinks)
        if not materialized_sinks:
            raise ValueError("provide at least one event sink")
        if not all(isinstance(sink, EventSink) for sink in materialized_sinks):
            raise TypeError("every sink must provide write(record)")

        self._formatter = formatter
        self._sinks = materialized_sinks

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """Return the collaborators without exposing mutable configuration."""
        return self._sinks

    def publish(self, event: Event) -> str:
        """Publish one event and return the formatted record."""
        if not isinstance(event, Event):
            raise TypeError("event must be an Event")

        record = self._formatter.format(event)
        if not isinstance(record, str) or not record:
            raise ValueError("formatter must return non-empty text")

        for sink in self._sinks:
            sink.write(record)
        return record

    def publish_many(self, events: Iterable[Event]) -> tuple[str, ...]:
        """Publish a possibly one-shot iterable in its original order."""
        return tuple(self.publish(event) for event in events)


# Practice exercise solutions


@runtime_checkable
class Named(Protocol):
    """Solution 1 contract: expose a readable string name."""

    @property
    def name(self) -> str:
        """Return the object's name."""


def sorted_names(items: Iterable[Named]) -> tuple[str, ...]:
    """Solution 1: normalize and sort names from unrelated object types."""
    names: list[str] = []
    for item in items:
        if not isinstance(item, Named):
            raise TypeError("every item must provide a name")
        names.append(_clean_text(item.name, "name"))
    return tuple(sorted(names, key=str.casefold))


class PrefixFormatter:
    """Solution 2: adapt any formatter by decorating its result."""

    def __init__(self, prefix: str, formatter: EventFormatter) -> None:
        self._prefix = _clean_text(prefix, "prefix")
        if not isinstance(formatter, EventFormatter):
            raise TypeError("formatter must provide format(event)")
        self._formatter = formatter

    def format(self, event: Event, /) -> str:
        return f"{self._prefix} {self._formatter.format(event)}"


class CallbackSink:
    """Solution 3: adapt a callable to the EventSink interface."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callback = callback

    def write(self, record: str, /) -> None:
        self._callback(_clean_text(record, "record"))


@dataclass(frozen=True, slots=True)
class Course:
    """Unrelated type that satisfies Named through its name property."""

    name: str


class _MissingFormat:
    """Test helper that deliberately does not satisfy EventFormatter."""


class _BadSink:
    """Test helper that deliberately does not satisfy EventSink."""


def run_self_checks() -> None:
    """Check structural interfaces, adapters, ordering, and edge cases."""
    event = Event(
        " Course.Started ",
        " Day 11 ",
        (("Student", "Asha"), ("Track", "Python")),
    )
    assert event.kind == "course.started"
    assert event.details == (("student", "Asha"), ("track", "Python"))

    line_formatter = LineFormatter()
    json_formatter = JsonFormatter()
    assert isinstance(line_formatter, EventFormatter)
    assert isinstance(ListSink(), EventSink)
    assert line_formatter.format(event) == (
        "course.started: Day 11 (student=Asha, track=Python)"
    )
    assert json_formatter.format(event) == (
        '{"details":{"student":"Asha","track":"Python"},'
        '"kind":"course.started","message":"Day 11"}'
    )

    first_sink = ListSink()
    forwarded: list[str] = []
    publisher = EventPublisher(
        PrefixFormatter("[course]", line_formatter),
        [first_sink, CallbackSink(forwarded.append)],
    )
    record = publisher.publish(event)
    assert record == (
        "[course] course.started: Day 11 "
        "(student=Asha, track=Python)"
    )
    assert first_sink.records == (record,)
    assert forwarded == [record]

    more_records = publisher.publish_many(
        Event("lesson.completed", title)
        for title in ("Protocols", "Duck typing")
    )
    assert len(more_records) == 2
    assert first_sink.records[-1].endswith("Duck typing")

    assert sorted_names(
        [Course(" typing "), Course("Async"), Course("dataclasses")]
    ) == ("Async", "dataclasses", "typing")

    invalid_calls = (
        lambda: Event("", "message"),
        lambda: Event("kind", "message", (("key", "one"), ("KEY", "two"))),
        lambda: Event("kind", "message", (("only-one-item",),)),  # type: ignore[arg-type]
        lambda: EventPublisher(_MissingFormat(), [ListSink()]),  # type: ignore[arg-type]
        lambda: EventPublisher(LineFormatter(), []),
        lambda: EventPublisher(LineFormatter(), [_BadSink()]),  # type: ignore[list-item]
        lambda: EventPublisher(LineFormatter(), [ListSink()]).publish("bad"),  # type: ignore[arg-type]
        lambda: sorted_names([object()]),  # type: ignore[list-item]
        lambda: PrefixFormatter("", LineFormatter()),
        lambda: CallbackSink(None),  # type: ignore[arg-type]
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

    sink = ListSink()
    publisher = EventPublisher(JsonFormatter(), [sink])
    publisher.publish(
        Event(
            "lesson.completed",
            "Protocols and duck typing",
            (("day", "11"), ("status", "passed")),
        )
    )

    print("Stored record:", sink.records[0])
    print("Structural formatter:", isinstance(LineFormatter(), EventFormatter))
    print(
        "Sorted names:",
        sorted_names([Course("Generators"), Course("Typing")]),
    )
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
