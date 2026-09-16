"""Day 15: Decorators, wrappers, and reusable instrumentation.

Learning goals
--------------
1. Explain that a decorator receives a callable and returns a callable.
2. Preserve wrapped function metadata with functools.wraps.
3. Add logging, timing, validation, and retry behavior without editing the
   business function itself.
4. Stack decorators deliberately and understand which wrapper runs first.
5. Keep decorator configuration explicit, testable, and free of hidden global
   side effects.

Teaching notes
--------------
- ``@decorator`` is syntax for rebinding a function:
  ``work = decorator(work)``. The decorated name may refer to a wrapper.
- A wrapper should accept the original arguments, call the original function,
  and return its result. ``functools.wraps`` copies metadata such as
  ``__name__`` and ``__doc__`` and exposes ``__wrapped__`` for introspection.
- ParamSpec and TypeVar let a decorator preserve the wrapped callable's
  argument and return types for static checkers.
- A decorator factory is a function that receives configuration and returns a
  decorator. ``retry(max_attempts=3)`` is a decorator factory.
- Decorator order matters. In ``@outer`` above ``@inner``, Python evaluates
  ``outer(inner(function))``. Write a tiny trace when the order is unclear.
- Instrumentation should report useful facts without leaking secrets. Log
  operation names and durations, not passwords, tokens, or full payment data.
- Retry only failures that are safe to retry. Limit attempts, avoid retrying
  validation errors, and make idempotency expectations explicit.
- A decorator can be a clean cross-cutting boundary, but too many wrappers can
  hide control flow. Prefer a plain helper when behavior is local and obvious.
- Decorators run at definition/import time. Keep registration deterministic and
  avoid network calls or expensive work while a module is imported.

Run this file with Python 3.10+ to execute deterministic examples and checks.
No third-party package is required.

Practice exercises
------------------
1. Write ``require_keyword`` so a decorator rejects calls missing a required
   keyword argument and preserves the wrapped function's metadata.
2. Write ``count_calls`` that returns a wrapped function and exposes a read-only
   ``calls`` attribute for tests.
3. Write ``compose_decorators(*decorators)`` and document the order in which
   the supplied decorators are applied.

Solutions appear below the main example.

Expert challenge: instrumented command registry
-----------------------------------------------
Build a command registry using a ``@command(name)`` decorator. Add a dispatcher
that validates command names, records duration and success/failure metadata,
and converts only expected domain exceptions into a structured error result.
Keep command functions independent of the registry so they remain easy to test.

Solution guidance
-----------------
1. Store a callable and its public name in an explicit dictionary.
2. Use ``functools.wraps`` for observability and ``inspect.signature`` only at
   the boundary where argument validation is actually needed.
3. Inject a clock and log sink into instrumentation for deterministic tests.
4. Never evaluate command text with ``eval`` or import arbitrary module names.
5. Define whether a failed command is retried, reported, or allowed to raise.
6. Test duplicate names, unknown commands, malformed arguments, expected
   domain failures, wrapper metadata, and decorator ordering.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
import time
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

Clock = Callable[[], float]
LogSink = Callable[[str], None]


def _clean_text(value: str, field_name: str) -> str:
    """Return stripped, non-empty text with a useful validation error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _positive_int(value: int, field_name: str) -> int:
    """Reject Booleans and non-positive integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


class CommandError(Exception):
    """Expected failure raised by a command at the application boundary."""


@dataclass(frozen=True, slots=True)
class CallEvent:
    """Safe instrumentation data; it intentionally excludes arguments."""

    function_name: str
    elapsed_seconds: float
    succeeded: bool
    error_type: str | None = None


def trace_calls(
    sink: LogSink,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record call start and outcome without exposing argument values."""
    if not callable(sink):
        raise TypeError("sink must be callable")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if not callable(function):
            raise TypeError("function must be callable")

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sink(f"call.start:{function.__name__}")
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                sink(f"call.error:{function.__name__}:{type(error).__name__}")
                raise
            sink(f"call.success:{function.__name__}")
            return result

        return wrapper

    return decorate


def measure_calls(
    events: list[CallEvent],
    *,
    clock: Clock = time.perf_counter,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record duration and outcome, with an injectable monotonic clock."""
    if not isinstance(events, list):
        raise TypeError("events must be a list")
    if not callable(clock):
        raise TypeError("clock must be callable")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = clock()
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                events.append(
                    CallEvent(
                        function.__name__,
                        max(0.0, clock() - started),
                        False,
                        type(error).__name__,
                    )
                )
                raise
            events.append(
                CallEvent(
                    function.__name__,
                    max(0.0, clock() - started),
                    True,
                )
            )
            return result

        return wrapper

    return decorate


def retry(
    *,
    max_attempts: int = 3,
    retryable: tuple[type[Exception], ...] = (TimeoutError,),
    on_retry: Callable[[str, int, Exception], None] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry selected exceptions with a bounded number of attempts.

    This decorator performs no sleeping. A production caller can notify a
    scheduler or inject a backoff policy rather than blocking a worker here.
    """
    attempts = _positive_int(max_attempts, "max_attempts")
    if not retryable or not all(
        isinstance(error_type, type) and issubclass(error_type, Exception)
        for error_type in retryable
    ):
        raise TypeError("retryable must contain exception types")
    if on_retry is not None and not callable(on_retry):
        raise TypeError("on_retry must be callable")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, attempts + 1):
                try:
                    return function(*args, **kwargs)
                except retryable as error:
                    if attempt == attempts:
                        raise
                    if on_retry is not None:
                        on_retry(function.__name__, attempt, error)
            raise AssertionError("retry loop must return or raise")

        return wrapper

    return decorate


def require_keyword(
    keyword: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Solution 1: require a named keyword in each call."""
    required_name = _clean_text(keyword, "keyword")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if required_name not in kwargs:
                raise TypeError(f"missing required keyword: {required_name}")
            return function(*args, **kwargs)

        return wrapper

    return decorate


def count_calls(
    function: Callable[P, R],
) -> Callable[P, R]:
    """Solution 2: wrap a function and expose a test-friendly call count."""
    if not callable(function):
        raise TypeError("function must be callable")

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        setattr(wrapper, "calls", getattr(wrapper, "calls") + 1)
        return function(*args, **kwargs)

    setattr(wrapper, "calls", 0)
    return wrapper


def compose_decorators(
    *decorators: Callable[[Callable[P, R]], Callable[P, R]],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Solution 3: apply decorators left to right in written order.

    ``compose_decorators(outer, inner)(function)`` is equivalent to
    ``outer(inner(function))``.
    """
    if not decorators or not all(callable(decorator) for decorator in decorators):
        raise ValueError("provide one or more callable decorators")

    def apply(function: Callable[P, R]) -> Callable[P, R]:
        decorated = function
        for decorator in reversed(decorators):
            decorated = decorator(decorated)
        return decorated

    return apply


COMMANDS: dict[str, Callable[..., object]] = {}


def command(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register a callable under a safe, explicit command name."""
    command_name = _clean_text(name, "command name").casefold()
    if not command_name.replace("-", "_").isidentifier():
        raise ValueError("command name must be an identifier-like name")

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if command_name in COMMANDS:
            raise ValueError(f"duplicate command: {command_name}")
        COMMANDS[command_name] = function
        return function

    return decorate


@command("add")
def add_numbers(left: int, right: int) -> int:
    """Return the sum of two integers."""
    if isinstance(left, bool) or isinstance(right, bool):
        raise CommandError("Boolean values are not valid integers")
    return left + right


@command("divide")
def divide_numbers(left: float, right: float) -> float:
    """Divide two finite numbers."""
    if right == 0:
        raise CommandError("cannot divide by zero")
    return left / right


def dispatch(
    name: str,
    *args: object,
    registry: Mapping[str, Callable[..., object]] = COMMANDS,
    **kwargs: object,
) -> object:
    """Dispatch only a registered callable and preserve expected failures."""
    command_name = _clean_text(name, "command name").casefold()
    try:
        function = registry[command_name]
    except KeyError as error:
        raise CommandError(f"unknown command: {command_name}") from error
    if not callable(function):
        raise TypeError("registry value must be callable")
    try:
        return function(*args, **kwargs)
    except CommandError:
        raise
    except TypeError as error:
        raise CommandError(f"invalid arguments for command: {command_name}") from error
def run_self_checks() -> None:
    """Check metadata, decorator order, retries, and important edge cases."""
    events: list[CallEvent] = []
    clock_values = iter((10.0, 10.25, 20.0, 20.5))
    measured = measure_calls(events, clock=lambda: next(clock_values))

    @measured
    def square(number: int) -> int:
        """Return a square."""
        return number * number

    assert square.__name__ == "square"
    assert square.__doc__ == "Return a square."
    assert square(4) == 16
    assert events == [CallEvent("square", 0.25, True)]

    messages: list[str] = []

    @trace_calls(messages.append)
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert greet("Asha") == "Hello, Asha!"
    assert messages == ["call.start:greet", "call.success:greet"]

    retry_events: list[tuple[str, int]] = []
    attempts = {"count": 0}

    @retry(
        max_attempts=3,
        on_retry=lambda name, attempt, error: retry_events.append((name, attempt)),
    )
    def unstable() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary")
        return "ready"

    assert unstable() == "ready"
    assert retry_events == [("unstable", 1), ("unstable", 2)]
    assert attempts["count"] == 3

    called = count_calls(lambda value: value + 1)
    assert called(2) == 3
    assert called(4) == 5
    assert called.calls == 2

    @require_keyword("confirm")
    def publish(*, confirm: bool) -> str:
        return "published" if confirm else "blocked"

    assert publish(confirm=True) == "published"
    try:
        publish()
    except TypeError as error:
        assert "confirm" in str(error)
    else:
        raise AssertionError("missing keyword must raise TypeError")

    order: list[str] = []

    def mark(label: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
        def decorate(function: Callable[[], str]) -> Callable[[], str]:
            @wraps(function)
            def wrapper() -> str:
                order.append(f"{label}.before")
                result = function()
                order.append(f"{label}.after")
                return result

            return wrapper

        return decorate

    decorated = compose_decorators(mark("outer"), mark("inner"))(
        lambda: "done"
    )
    assert decorated() == "done"
    assert order == [
        "outer.before",
        "inner.before",
        "inner.after",
        "outer.after",
    ]

    assert dispatch("add", 2, 3) == 5
    assert dispatch("divide", 9, 2) == 4.5
    try:
        dispatch("divide", 9, 0)
    except CommandError as error:
        assert "zero" in str(error)
    else:
        raise AssertionError("domain failure must remain a CommandError")
    try:
        dispatch("missing")
    except CommandError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown commands must be rejected")

    invalid_calls = (
        lambda: trace_calls(None),  # type: ignore[arg-type]
        lambda: measure_calls([], clock=None),  # type: ignore[arg-type]
        lambda: retry(max_attempts=0),
        lambda: retry(retryable=(str,)),  # type: ignore[arg-type]
        lambda: require_keyword(""),
        lambda: count_calls(None),  # type: ignore[arg-type]
        lambda: compose_decorators(),
        lambda: command("not valid"),
        lambda: command("add")(lambda: None),
        lambda: dispatch("divide", "9", "2"),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (CommandError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without external services."""
    run_self_checks()

    events: list[CallEvent] = []
    clock_values = iter((100.0, 100.125))

    @measure_calls(events, clock=lambda: next(clock_values))
    @trace_calls(print)
    def total_with_tax(price: float, tax_rate: float = 0.13) -> float:
        """Calculate a price after tax."""
        return round(price * (1 + tax_rate), 2)

    print("Total:", total_with_tax(100.0))
    print("Instrumentation:", events)
    print("Commands:", sorted(COMMANDS))
    print("Dispatch result:", dispatch("add", 8, 5))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
