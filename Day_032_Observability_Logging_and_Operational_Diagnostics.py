"""Day 32: Observability, structured logging, and operational diagnostics.

Learning goals
--------------
1. Distinguish logs, metrics, and traces by the questions each signal answers.
2. Emit structured JSON logs with stable event names and UTC timestamps.
3. Propagate request context safely with context variables.
4. Redact secrets and keep high-cardinality values out of metric dimensions.
5. Test success, failure, cleanup, and diagnostic summaries deterministically.

Teaching notes
--------------
- Observability is the ability to explain a system from evidence it emits. Logs
  describe discrete events, metrics summarize trends, and traces connect work
  across boundaries. One signal does not replace the others.
- Prefer stable event names such as "task.create.completed" over prose that is
  difficult to query. Put changing values in structured fields.
- Configure handlers at an application entry point. Library modules should use
  named loggers without calling basicConfig or adding global handlers.
- Context variables isolate request metadata across threads and asyncio tasks.
  Always reset a context token in a finally block so one request cannot leak
  identifiers into another.
- Treat logs as an information boundary. Redact credentials by key, avoid raw
  request bodies, and do not assume an exception message is safe.
- Measure elapsed work with a monotonic clock. Use wall-clock UTC only for event
  timestamps. Inject clocks when deterministic tests need exact durations.
- Metric dimensions must have bounded cardinality. Route templates and outcome
  categories are useful; request IDs, user IDs, and raw URLs are not.
- Log failures once at the boundary that can add useful context. Preserve the
  original exception for callers while emitting a safe error category.
- Operational diagnostics should help answer: what failed, where, how often,
  and for how long? They should not expose secrets or personal data.

Run this file with Python 3.10+.
It uses only the standard library, opens no sockets, and writes no files.

Practice exercises
------------------
1. Add a sampling policy for successful debug events while always keeping
   warnings and errors. Inject the sampler so tests do not use randomness.
2. Add fixed latency buckets to Diagnostics and export cumulative counts without
   placing request IDs or raw paths in metric labels.
3. Write an asyncio test that binds different request IDs in concurrent tasks
   and proves that each captured event retains the correct context.

Expert challenge: observable service boundary
---------------------------------------------
Wrap the Day 31 task API with middleware that accepts or creates a validated
request ID, records one completion event, measures duration, and publishes
bounded route/status metrics. Add a queue-based log handler with a documented
backpressure policy, health and readiness diagnostics, trace-header propagation,
and tests for concurrency, cancellation, redaction, handler failure, and clean
shutdown. Define retention and access rules for operational data.

Solution guidance
-----------------
1. Make sampling a callable that receives level and event name. Apply it before
   constructing expensive fields; never sample away security or failure events.
2. Store bucket boundaries as a sorted immutable tuple. Increment only known
   operation/outcome keys and expose request IDs only in logs or traces.
3. ContextVar values are copied when asyncio tasks are created. Bind context
   inside each task, reset the token in finally, and assert parsed JSON events
   rather than depending on line order.
4. Keep observability adapters outside domain code. Tests should inject clocks,
   streams, and ID factories and should restore logger state after each case.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
import logging
import math
import re
from typing import TypeVar


T = TypeVar("T")

_EVENT_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SECRET_KEYS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "secret",
        "setcookie",
        "token",
    }
)
_STANDARD_LEVELS = frozenset(
    {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
)


def _clean_event(value: str, *, field: str = "event") -> str:
    """Validate a stable, machine-queryable event or operation name."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not _EVENT_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"{field} must use lowercase words separated by '.', '_', or '-'"
        )
    return cleaned


def _clean_request_id(value: str) -> str:
    """Validate a bounded opaque request identifier without changing it."""
    if not isinstance(value, str):
        raise TypeError("request_id must be text")
    cleaned = value.strip()
    if not _REQUEST_ID_PATTERN.fullmatch(cleaned):
        raise ValueError("request_id contains unsupported characters or length")
    return cleaned


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _safe_value(value: object, *, depth: int = 0) -> object:
    """Convert a value to bounded JSON-safe data without calling arbitrary repr."""
    if depth >= 3:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE]"
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        for key, item in list(value.items())[:20]:
            if not isinstance(key, str):
                raise TypeError("diagnostic field names must be text")
            if _normalized_key(key) in _SECRET_KEYS:
                safe_mapping[key] = "[REDACTED]"
            else:
                safe_mapping[key] = _safe_value(item, depth=depth + 1)
        return safe_mapping
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth=depth + 1) for item in value[:20]]
    return f"<{type(value).__name__}>"


def redact_fields(fields: Mapping[str, object]) -> dict[str, object]:
    """Return a shallow field mapping with recursive safe values."""
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping")
    redacted = _safe_value(fields)
    if not isinstance(redacted, dict):
        raise AssertionError("mapping conversion must produce a dictionary")
    return redacted


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Low-cardinality operation plus a high-cardinality correlation ID."""

    request_id: str
    operation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _clean_request_id(self.request_id))
        object.__setattr__(
            self,
            "operation",
            _clean_event(self.operation, field="operation"),
        )


_CURRENT_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "day32_request_context",
    default=None,
)


@contextmanager
def bind_request_context(
    request_id: str,
    operation: str,
) -> Iterator[RequestContext]:
    """Bind context for one unit of work and always restore the previous value."""
    context = RequestContext(request_id=request_id, operation=operation)
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


class RequestContextFilter(logging.Filter):
    """Snapshot the current request context onto each LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _CURRENT_CONTEXT.get()
        record.request_id = None if context is None else context.request_id
        record.operation = None if context is None else context.operation
        return True


class JsonFormatter(logging.Formatter):
    """Format a small, stable JSON event schema."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        document: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event_name", record.getMessage()),
        }

        request_id = getattr(record, "request_id", None)
        operation = getattr(record, "operation", None)
        if request_id is not None:
            document["request_id"] = request_id
        if operation is not None:
            document["operation"] = operation

        fields = getattr(record, "event_fields", {})
        if fields:
            document["fields"] = fields

        if record.exc_info and record.exc_info[0] is not None:
            document["error"] = {
                "type": record.exc_info[0].__name__,
                "category": "operation_failure",
            }

        return json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    fields: Mapping[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    """Emit one structured event while keeping exception messages out of logs."""
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    if isinstance(level, bool) or level not in _STANDARD_LEVELS:
        raise ValueError("level must be a standard logging level")
    event_name = _clean_event(event)
    safe_fields = redact_fields({} if fields is None else fields)
    exc_info = None
    if error is not None:
        exc_info = (type(error), error, error.__traceback__)
    logger.log(
        level,
        event_name,
        extra={"event_name": event_name, "event_fields": safe_fields},
        exc_info=exc_info,
    )


@contextmanager
def capture_json_logs(name: str) -> Iterator[tuple[logging.Logger, StringIO]]:
    """Create an isolated JSON logger and restore all prior state afterward."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("logger name must be non-empty text")
    logger = logging.getLogger(name)
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    old_disabled = logger.disabled

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())

    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.disabled = False
    try:
        yield logger, stream
    finally:
        handler.close()
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        logger.disabled = old_disabled


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """One bounded operation/outcome metric series."""

    operation: str
    outcome: str
    count: int
    total_duration_ms: float
    max_duration_ms: float

    @property
    def average_duration_ms(self) -> float:
        return round(self.total_duration_ms / self.count, 3)

    def as_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "outcome": self.outcome,
            "count": self.count,
            "average_duration_ms": self.average_duration_ms,
            "max_duration_ms": self.max_duration_ms,
        }


class Diagnostics:
    """In-memory metrics with an explicit, bounded operation vocabulary."""

    def __init__(self, allowed_operations: Iterable[str]) -> None:
        operations = frozenset(
            _clean_event(value, field="operation") for value in allowed_operations
        )
        if not operations:
            raise ValueError("at least one allowed operation is required")
        self._allowed_operations = operations
        self._series: dict[tuple[str, str], list[float]] = {}

    def require_operation(self, operation: str) -> str:
        cleaned = _clean_event(operation, field="operation")
        if cleaned not in self._allowed_operations:
            raise ValueError(f"unregistered operation: {cleaned}")
        return cleaned

    def observe(self, operation: str, outcome: str, duration_ms: float) -> None:
        clean_operation = self.require_operation(operation)
        if outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or not math.isfinite(float(duration_ms))
            or duration_ms < 0
        ):
            raise ValueError("duration_ms must be a finite non-negative number")

        key = (clean_operation, outcome)
        count, total, maximum = self._series.get(key, [0.0, 0.0, 0.0])
        measured = float(duration_ms)
        self._series[key] = [
            count + 1,
            total + measured,
            max(maximum, measured),
        ]

    def snapshot(self) -> tuple[MetricSnapshot, ...]:
        snapshots = []
        for (operation, outcome), (count, total, maximum) in sorted(
            self._series.items()
        ):
            snapshots.append(
                MetricSnapshot(
                    operation=operation,
                    outcome=outcome,
                    count=int(count),
                    total_duration_ms=round(total, 3),
                    max_duration_ms=round(maximum, 3),
                )
            )
        return tuple(snapshots)


def _elapsed_ms(start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("clock must return finite numbers")
    if end < start:
        raise ValueError("monotonic clock moved backwards")
    return round((end - start) * 1_000, 3)


def run_observed(
    logger: logging.Logger,
    diagnostics: Diagnostics,
    operation: str,
    action: Callable[[], T],
    *,
    clock: Callable[[], float],
) -> T:
    """Run one action, preserve failures, and emit bounded diagnostics."""
    clean_operation = diagnostics.require_operation(operation)
    if not callable(action) or not callable(clock):
        raise TypeError("action and clock must be callable")

    start = float(clock())
    log_event(
        logger,
        logging.INFO,
        f"{clean_operation}.started",
    )
    try:
        result = action()
    except Exception as error:
        duration_ms = _elapsed_ms(start, float(clock()))
        diagnostics.observe(clean_operation, "failure", duration_ms)
        log_event(
            logger,
            logging.ERROR,
            f"{clean_operation}.failed",
            fields={"duration_ms": duration_ms, "reason": "runtime_error"},
            error=error,
        )
        raise

    duration_ms = _elapsed_ms(start, float(clock()))
    diagnostics.observe(clean_operation, "success", duration_ms)
    log_event(
        logger,
        logging.INFO,
        f"{clean_operation}.completed",
        fields={"duration_ms": duration_ms},
    )
    return result


def diagnostic_report(diagnostics: Diagnostics) -> dict[str, object]:
    """Build a JSON-ready operational summary from bounded metric series."""
    snapshots = diagnostics.snapshot()
    total = sum(item.count for item in snapshots)
    failures = sum(item.count for item in snapshots if item.outcome == "failure")
    return {
        "operations": [item.as_dict() for item in snapshots],
        "total": total,
        "failures": failures,
        "failure_rate": 0.0 if total == 0 else round(failures / total, 3),
    }


def _json_lines(stream: StringIO) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if line.strip()
    )


def run_self_checks() -> None:
    """Verify schemas, redaction, context cleanup, metrics, and failures."""
    diagnostics = Diagnostics(("task.create", "task.list"))

    with capture_json_logs("course.day32.selfcheck") as (logger, stream):
        success_clock = iter((10.0, 10.025)).__next__
        with bind_request_context("req-32_A", "task.create"):
            result = run_observed(
                logger,
                diagnostics,
                "task.create",
                lambda: {"task_id": "task-001"},
                clock=success_clock,
            )
            assert result == {"task_id": "task-001"}
            log_event(
                logger,
                logging.INFO,
                "task.create.audit",
                fields={
                    "authorization": "Bearer do-not-log",
                    "nested": {"api_key": "hidden", "attempt": 1},
                    "payload": object(),
                    "temperature": float("nan"),
                },
            )

        assert _CURRENT_CONTEXT.get() is None

        def fail() -> None:
            raise TimeoutError("message may contain unsafe upstream details")

        failure_clock = iter((20.0, 20.010)).__next__
        try:
            with bind_request_context("req-32_B", "task.list"):
                run_observed(
                    logger,
                    diagnostics,
                    "task.list",
                    fail,
                    clock=failure_clock,
                )
        except TimeoutError:
            pass
        else:
            raise AssertionError("the original operation failure must propagate")

        events = _json_lines(stream)

    assert len(events) == 5
    assert events[0]["event"] == "task.create.started"
    assert events[0]["request_id"] == "req-32_A"
    assert str(events[0]["timestamp"]).endswith("Z")
    assert events[1]["fields"] == {"duration_ms": 25.0}
    audit_fields = events[2]["fields"]
    assert isinstance(audit_fields, dict)
    assert audit_fields["authorization"] == "[REDACTED]"
    assert audit_fields["nested"]["api_key"] == "[REDACTED]"
    assert audit_fields["payload"] == "<object>"
    assert audit_fields["temperature"] == "[NON_FINITE]"
    assert events[-1]["level"] == "ERROR"
    assert events[-1]["error"] == {
        "category": "operation_failure",
        "type": "TimeoutError",
    }
    assert "message may contain" not in json.dumps(events[-1])

    report = diagnostic_report(diagnostics)
    assert report["total"] == 2
    assert report["failures"] == 1
    assert report["failure_rate"] == 0.5
    snapshots = diagnostics.snapshot()
    assert snapshots[0].operation == "task.create"
    assert snapshots[0].average_duration_ms == 25.0
    assert snapshots[1].max_duration_ms == 10.0

    invalid_calls = (
        lambda: RequestContext("", "task.create"),
        lambda: RequestContext("spaces are invalid", "task.create"),
        lambda: RequestContext("req-1", "Task Create"),
        lambda: Diagnostics(()),
        lambda: diagnostics.observe("unknown.operation", "success", 1.0),
        lambda: diagnostics.observe("task.create", "unknown", 1.0),
        lambda: diagnostics.observe("task.create", "success", -1.0),
        lambda: redact_fields({1: "not text"}),  # type: ignore[dict-item]
        lambda: log_event(logging.getLogger("x"), logging.NOTSET, "valid.event"),
        lambda: _elapsed_ms(2.0, 1.0),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid diagnostic input must fail clearly")


def main() -> None:
    """Run a deterministic local observability demonstration."""
    run_self_checks()
    diagnostics = Diagnostics(("task.create", "task.list"))

    with capture_json_logs("course.day32.demo") as (logger, stream):
        with bind_request_context("demo-request-32", "task.create"):
            clock = iter((100.0, 100.012)).__next__
            task_id = run_observed(
                logger,
                diagnostics,
                "task.create",
                lambda: "task-032",
                clock=clock,
            )
            log_event(
                logger,
                logging.INFO,
                "task.create.result",
                fields={"task_id": task_id, "token": "never-print-this"},
            )

        print("Structured events:")
        print(stream.getvalue().rstrip())

    print("Diagnostic report:")
    print(json.dumps(diagnostic_report(diagnostics), indent=2, sort_keys=True))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
