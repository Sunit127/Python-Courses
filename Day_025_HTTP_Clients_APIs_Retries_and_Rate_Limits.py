"""Day 25: HTTP clients, APIs, retries, and rate limits.

Learning goals
--------------
1. Model HTTP requests and responses without coupling business logic to a library.
2. Build and validate API URLs, headers, JSON bodies, and timeouts safely.
3. Retry only transient failures when an operation is safe to repeat.
4. respect Retry-After responses and pace requests with a rate limiter.
5. Test network behavior deterministically with injected transports and clocks.

Teaching notes
--------------
- HTTP is a request-response protocol. A request has a method, URL, headers, and
  optional body. A response has a status code, headers, and optional body.
- A timeout is part of every production request. Without one, a stalled service
  can hold a worker forever. Response size limits also protect memory.
- Status codes have different meanings. A 2xx response succeeded; most 4xx
  responses require the caller to change something; 429 means rate limited; and
  some 5xx responses are transient. Treat the API contract as authoritative.
- Retry only failures documented as transient. GET, HEAD, PUT, DELETE, and
  OPTIONS are normally idempotent. Retrying POST or PATCH is unsafe unless the
  server documents an idempotency-key contract and receives a stable key.
- Exponential backoff gives a recovering service room to breathe. A Retry-After
  header should influence the delay, while a configured cap prevents an
  accidental unbounded sleep. Production clients usually add random jitter.
- Rate limiting and retries are separate policies. Every attempt consumes
  capacity, so acquire rate-limit permission before each attempt.
- Parse JSON only after checking the status, content type, encoding, and body
  size. Do not expose authorization headers, tokens, or raw response bodies in
  logs and exception messages.
- Inject the transport, clock, and sleeper. Tests can then simulate 429, 503,
  timeouts, bad JSON, and pacing without using a real network or waiting.

Run this file with Python 3.10+. It uses only the standard library. The runnable
example uses a scripted local transport and performs no network access.

Practice exercises
------------------
1. Implement redact_headers so logs keep header names but mask credentials.
2. Implement cursor pagination with a page limit and repeated-cursor detection.
3. Parse rate-limit headers and calculate when a caller must wait.

Solutions appear below the client implementation.

Expert challenge: production API ingestion service
---------------------------------------------------
Build an ingestion command that reads cursor-paginated records from a documented
HTTPS API and stores them transactionally. Add stable request IDs, schema
validation, bounded retries, rate-limit awareness, checkpointed cursors, and
structured metrics. A restart must resume safely without duplicating records.

Solution guidance
-----------------
1. Separate configuration, transport, API client, schema validation, persistence,
   and orchestration behind narrow interfaces.
2. Store the last committed cursor in the same database transaction as the
   records it represents. Use a unique remote record ID for deduplication.
3. Retry only transient transport failures, 429, and documented 5xx responses.
   Use server-supported idempotency keys for any retried write.
4. Bound attempts, elapsed time, response bytes, pages, and records. Apply both
   connect/read timeouts in libraries that expose them.
5. Redact secrets, record latency and attempt counts, and keep safe request IDs.
   Never log authorization values or an entire untrusted response body.
6. Test empty pages, malformed JSON, schema drift, a repeated cursor, 429 with
   Retry-After, permanent 4xx, exhausted retries, and a crash before commit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen


JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
QueryValue = str | int | float | bool


def _clean_text(value: str, *, field_name: str) -> str:
    """Return stripped non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _positive_float(value: float, *, field_name: str) -> float:
    """Return a finite positive float and reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def _positive_int(value: int, *, field_name: str) -> int:
    """Return a positive integer and reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


class ApiClientError(Exception):
    """Base class for expected API client failures."""


class TransportError(ApiClientError):
    """The request could not produce an HTTP response."""


class ResponseValidationError(ApiClientError):
    """A response violated the client's size or JSON contract."""


class HttpStatusError(ApiClientError):
    """A non-success HTTP status that the client will not retry."""

    def __init__(self, status: int, method: str, url: str) -> None:
        self.status = status
        self.method = method
        self.url = url
        super().__init__(f"{method} request failed with HTTP {status}")


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """A bounded HTTP response independent of a particular HTTP library."""

    status: int
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""

    def __post_init__(self) -> None:
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("status must be an integer")
        if not 100 <= self.status <= 599:
            raise ValueError("status must be between 100 and 599")
        if not isinstance(self.body, bytes):
            raise TypeError("body must be bytes")
        for name, value in self.headers:
            _clean_text(name, field_name="header name")
            if not isinstance(value, str):
                raise TypeError("header values must be text")

    def header(self, name: str) -> str | None:
        """Return the last matching header value, ignoring name case."""
        target = _clean_text(name, field_name="header name").casefold()
        for header_name, value in reversed(self.headers):
            if header_name.casefold() == target:
                return value
        return None


class HttpTransport(Protocol):
    """The smallest transport interface needed by ApiClient."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        """Send one request or raise TransportError."""


class UrllibTransport:
    """A standard-library transport with bounded response reads."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        request = Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                response_body = response.read(max_response_bytes + 1)
                if len(response_body) > max_response_bytes:
                    raise ResponseValidationError("response body is too large")
                return HttpResponse(
                    status=response.status,
                    headers=tuple(response.headers.items()),
                    body=response_body,
                )
        except HTTPError as error:
            response_body = error.read(max_response_bytes + 1)
            if len(response_body) > max_response_bytes:
                raise ResponseValidationError("error response body is too large")
            return HttpResponse(
                status=error.code,
                headers=tuple(error.headers.items()),
                body=response_body,
            )
        except (URLError, TimeoutError, OSError) as error:
            raise TransportError("request transport failed") from error


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff for documented transient failures."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )

    def __post_init__(self) -> None:
        _positive_int(self.max_attempts, field_name="max_attempts")
        _positive_float(
            self.base_delay_seconds,
            field_name="base_delay_seconds",
        )
        _positive_float(
            self.max_delay_seconds,
            field_name="max_delay_seconds",
        )
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base delay cannot exceed maximum delay")
        if any(
            isinstance(status, bool)
            or not isinstance(status, int)
            or not 100 <= status <= 599
            for status in self.retry_statuses
        ):
            raise ValueError("retry_statuses must contain HTTP status integers")

    def delay_for(
        self,
        failed_attempt: int,
        *,
        retry_after_seconds: float | None = None,
    ) -> float:
        """Return a capped delay after a one-based failed attempt."""
        _positive_int(failed_attempt, field_name="failed_attempt")
        exponential = self.base_delay_seconds * (2 ** (failed_attempt - 1))
        delay = exponential
        if retry_after_seconds is not None:
            if (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, (int, float))
            ):
                raise TypeError("retry_after_seconds must be a number")
            retry_after = float(retry_after_seconds)
            if not math.isfinite(retry_after) or retry_after < 0:
                raise ValueError("retry_after_seconds must be finite and non-negative")
            delay = max(delay, retry_after)
        return min(delay, self.max_delay_seconds)


def parse_retry_after(
    value: str | None,
    *,
    wall_clock: Callable[[], float] = time.time,
) -> float | None:
    """Parse Retry-After delta seconds or an HTTP date."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        return float(cleaned)
    try:
        target = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, target.timestamp() - wall_clock())


class RateLimiter(Protocol):
    """A synchronous rate limiter used before every attempt."""

    def acquire(self) -> None:
        """Wait until one request attempt may start."""


class NoRateLimiter:
    """A rate limiter that never waits."""

    def acquire(self) -> None:
        pass


class PacedRateLimiter:
    """Space requests for one synchronous caller by a minimum interval."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        rate = _positive_float(
            requests_per_second,
            field_name="requests_per_second",
        )
        if not callable(clock) or not callable(sleeper):
            raise TypeError("clock and sleeper must be callable")
        self._minimum_interval = 1.0 / rate
        self._clock = clock
        self._sleeper = sleeper
        self._next_allowed_at: float | None = None

    def acquire(self) -> None:
        now = self._clock()
        if self._next_allowed_at is not None and now < self._next_allowed_at:
            self._sleeper(self._next_allowed_at - now)
            now = self._clock()
        start = max(now, self._next_allowed_at or now)
        self._next_allowed_at = start + self._minimum_interval


class ApiClient:
    """A small JSON API client with explicit safety and retry policies."""

    _IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})
    _ALLOWED_METHODS = _IDEMPOTENT_METHODS | frozenset({"POST", "PATCH"})

    def __init__(
        self,
        base_url: str,
        *,
        transport: HttpTransport | None = None,
        retry_policy: RetryPolicy | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 1_000_000,
        sleeper: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
        allow_http: bool = False,
    ) -> None:
        cleaned_url = _clean_text(base_url, field_name="base_url")
        parts = urlsplit(cleaned_url)
        allowed_schemes = {"https", "http"} if allow_http else {"https"}
        if parts.scheme.casefold() not in allowed_schemes:
            raise ValueError("base_url must use an allowed HTTP scheme")
        if not parts.hostname or parts.username or parts.password:
            raise ValueError("base_url must have a host and no embedded credentials")
        if parts.query or parts.fragment:
            raise ValueError("base_url cannot contain a query or fragment")
        path = parts.path if parts.path.endswith("/") else parts.path + "/"
        self._base_url = parts._replace(path=path).geturl()
        self._transport = transport or UrllibTransport()
        self._retry_policy = retry_policy or RetryPolicy()
        self._rate_limiter = rate_limiter or NoRateLimiter()
        self._timeout = _positive_float(
            timeout_seconds,
            field_name="timeout_seconds",
        )
        self._max_response_bytes = _positive_int(
            max_response_bytes,
            field_name="max_response_bytes",
        )
        if not callable(sleeper) or not callable(wall_clock):
            raise TypeError("sleeper and wall_clock must be callable")
        self._sleeper = sleeper
        self._wall_clock = wall_clock

    def _build_url(
        self,
        endpoint: str,
        params: Mapping[str, QueryValue] | None,
    ) -> str:
        endpoint = _clean_text(endpoint, field_name="endpoint")
        parts = urlsplit(endpoint)
        if (
            parts.scheme
            or parts.netloc
            or parts.query
            or parts.fragment
            or endpoint.startswith("/")
            or "\\" in endpoint
            or any(character.isspace() for character in endpoint)
            or ".." in parts.path.split("/")
        ):
            raise ValueError("endpoint must be a safe relative path")
        url = urljoin(self._base_url, endpoint)
        if params:
            encoded: list[tuple[str, QueryValue]] = []
            for key, value in params.items():
                clean_key = _clean_text(key, field_name="query key")
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError("query values must be scalar")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("floating query values must be finite")
                encoded.append((clean_key, value))
            url = f"{url}?{urlencode(encoded)}"
        return url

    @staticmethod
    def _encode_json(value: JsonValue) -> bytes:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("json_body must contain valid JSON values") from error
        return text.encode("utf-8")

    @staticmethod
    def _decode_json(response: HttpResponse) -> JsonValue:
        if not response.body:
            return None
        content_type = (response.header("Content-Type") or "").split(";", 1)[0]
        normalized_type = content_type.strip().casefold()
        if not (
            normalized_type == "application/json"
            or normalized_type.endswith("+json")
        ):
            raise ResponseValidationError("successful response is not JSON")
        try:
            decoded = response.body.decode("utf-8")
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResponseValidationError("response contains invalid JSON") from error

    def request_json(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, QueryValue] | None = None,
        json_body: JsonValue = None,
        idempotency_key: str | None = None,
    ) -> JsonValue:
        """Send a JSON request and return the decoded JSON value."""
        method = _clean_text(method, field_name="method").upper()
        if method not in self._ALLOWED_METHODS:
            raise ValueError("unsupported HTTP method")
        url = self._build_url(endpoint, params)
        body = None if json_body is None else self._encode_json(json_body)
        headers = {
            "Accept": "application/json",
            "User-Agent": "Python-Courses-Day-025/1.0",
        }
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = _clean_text(
                idempotency_key,
                field_name="idempotency_key",
            )

        retryable_method = (
            method in self._IDEMPOTENT_METHODS or idempotency_key is not None
        )
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            self._rate_limiter.acquire()
            try:
                response = self._transport.request(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    timeout=self._timeout,
                    max_response_bytes=self._max_response_bytes,
                )
            except TransportError:
                if (
                    retryable_method
                    and attempt < self._retry_policy.max_attempts
                ):
                    self._sleeper(self._retry_policy.delay_for(attempt))
                    continue
                raise

            if len(response.body) > self._max_response_bytes:
                raise ResponseValidationError("response body is too large")
            if (
                response.status in self._retry_policy.retry_statuses
                and retryable_method
                and attempt < self._retry_policy.max_attempts
            ):
                retry_after = parse_retry_after(
                    response.header("Retry-After"),
                    wall_clock=self._wall_clock,
                )
                self._sleeper(
                    self._retry_policy.delay_for(
                        attempt,
                        retry_after_seconds=retry_after,
                    )
                )
                continue
            if not 200 <= response.status <= 299:
                raise HttpStatusError(response.status, method, url)
            return self._decode_json(response)

        raise AssertionError("retry loop ended unexpectedly")

    def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, QueryValue] | None = None,
    ) -> JsonValue:
        """Perform an idempotent GET request."""
        return self.request_json("GET", endpoint, params=params)

    def post_json(
        self,
        endpoint: str,
        json_body: JsonValue,
        *,
        idempotency_key: str | None = None,
    ) -> JsonValue:
        """Perform a POST; retry only when a stable key is supplied."""
        return self.request_json(
            "POST",
            endpoint,
            json_body=json_body,
            idempotency_key=idempotency_key,
        )


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """A safe request snapshot captured by ScriptedTransport."""

    method: str
    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes | None


class ScriptedTransport:
    """Return prepared results so client tests need no real network."""

    def __init__(self, results: Iterable[HttpResponse | Exception]) -> None:
        self._results = list(results)
        self.requests: list[RequestRecord] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        del timeout, max_response_bytes
        self.requests.append(
            RequestRecord(
                method=method,
                url=url,
                headers=tuple(sorted(headers.items())),
                body=body,
            )
        )
        if not self._results:
            raise AssertionError("scripted transport has no result left")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClock:
    """A deterministic monotonic clock and sleeper for examples."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep cannot be negative")
        self.sleeps.append(seconds)
        self.now += seconds


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Practice solution: mask common credential-bearing headers."""
    sensitive = {
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
    }
    redacted: dict[str, str] = {}
    for name, value in headers.items():
        clean_name = _clean_text(name, field_name="header name")
        if not isinstance(value, str):
            raise TypeError("header values must be text")
        redacted[clean_name] = (
            "[REDACTED]" if clean_name.casefold() in sensitive else value
        )
    return redacted


PageFetcher = Callable[[str | None], tuple[Sequence[object], str | None]]


def collect_cursor_pages(
    fetch_page: PageFetcher,
    *,
    start_cursor: str | None = None,
    max_pages: int = 100,
) -> tuple[object, ...]:
    """Practice solution: collect bounded pages and reject cursor loops."""
    if not callable(fetch_page):
        raise TypeError("fetch_page must be callable")
    limit = _positive_int(max_pages, field_name="max_pages")
    cursor = (
        None
        if start_cursor is None
        else _clean_text(start_cursor, field_name="start_cursor")
    )
    seen = set() if cursor is None else {cursor}
    collected: list[object] = []

    for _ in range(limit):
        items, next_cursor = fetch_page(cursor)
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise TypeError("page items must be a non-text sequence")
        collected.extend(items)
        if next_cursor is None:
            return tuple(collected)
        next_cursor = _clean_text(next_cursor, field_name="next_cursor")
        if next_cursor in seen:
            raise RuntimeError("API repeated a pagination cursor")
        seen.add(next_cursor)
        cursor = next_cursor
    raise RuntimeError("pagination exceeded max_pages")


@dataclass(frozen=True, slots=True)
class RateLimitWindow:
    """Practice solution: normalized quota state from response headers."""

    remaining: int
    reset_epoch: float
    wait_seconds: float


def parse_rate_limit_headers(
    headers: Mapping[str, str],
    *,
    now_epoch: float,
) -> RateLimitWindow:
    """Parse common remaining/reset headers and calculate required wait."""
    normalized = {name.casefold(): value for name, value in headers.items()}
    try:
        remaining = int(normalized["x-ratelimit-remaining"])
        reset_epoch = float(normalized["x-ratelimit-reset"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing or invalid rate-limit headers") from error
    if remaining < 0 or not math.isfinite(reset_epoch):
        raise ValueError("rate-limit values are out of range")
    now = float(now_epoch)
    if not math.isfinite(now):
        raise ValueError("now_epoch must be finite")
    wait = max(0.0, reset_epoch - now) if remaining == 0 else 0.0
    return RateLimitWindow(remaining, reset_epoch, wait)


def _json_response(
    value: JsonValue,
    *,
    status: int = 200,
    headers: tuple[tuple[str, str], ...] = (),
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=(("Content-Type", "application/json"),) + headers,
        body=json.dumps(value, allow_nan=False).encode("utf-8"),
    )


def run_self_checks() -> None:
    """Verify retries, safety boundaries, pacing, and practice solutions."""
    retry_clock = FakeClock()
    transport = ScriptedTransport(
        [
            HttpResponse(
                429,
                (("Content-Type", "application/json"), ("Retry-After", "2")),
                b'{"error":"slow down"}',
            ),
            _json_response({"lesson": 25, "topic": "HTTP"}),
        ]
    )
    client = ApiClient(
        "https://api.example.test/v1",
        transport=transport,
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.25,
            max_delay_seconds=3.0,
        ),
        sleeper=retry_clock.sleep,
        wall_clock=retry_clock.clock,
    )
    result = client.get_json(
        "lessons/25",
        params={"tag": "http client", "page": 1},
    )
    assert result == {"lesson": 25, "topic": "HTTP"}
    assert len(transport.requests) == 2
    assert transport.requests[0].url == (
        "https://api.example.test/v1/lessons/25?tag=http+client&page=1"
    )
    assert retry_clock.sleeps == [2.0]

    transient_transport = ScriptedTransport(
        [TransportError("temporary failure"), _json_response({"ok": True})]
    )
    transient_clock = FakeClock()
    transient_client = ApiClient(
        "https://api.example.test/",
        transport=transient_transport,
        retry_policy=RetryPolicy(max_attempts=2),
        sleeper=transient_clock.sleep,
    )
    assert transient_client.get_json("health") == {"ok": True}
    assert transient_clock.sleeps == [0.25]

    unsafe_transport = ScriptedTransport(
        [_json_response({"error": "busy"}, status=503), _json_response({"ok": True})]
    )
    unsafe_client = ApiClient(
        "https://api.example.test/",
        transport=unsafe_transport,
    )
    try:
        unsafe_client.post_json("events", {"name": "course"})
    except HttpStatusError as error:
        assert error.status == 503
    else:
        raise AssertionError("POST without an idempotency key must not retry")
    assert len(unsafe_transport.requests) == 1

    safe_transport = ScriptedTransport(
        [_json_response({"error": "busy"}, status=503), _json_response({"id": 7})]
    )
    safe_clock = FakeClock()
    safe_client = ApiClient(
        "https://api.example.test/",
        transport=safe_transport,
        retry_policy=RetryPolicy(max_attempts=2),
        sleeper=safe_clock.sleep,
    )
    assert safe_client.post_json(
        "events",
        {"name": "course"},
        idempotency_key="event-7",
    ) == {"id": 7}
    assert len(safe_transport.requests) == 2

    for invalid_endpoint in (
        "https://evil.example/data",
        "/absolute",
        "../escape",
        "items?secret=yes",
    ):
        try:
            client.get_json(invalid_endpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe endpoint must be rejected")

    invalid_responses = (
        HttpResponse(200, (("Content-Type", "text/html"),), b"not json"),
        HttpResponse(200, (("Content-Type", "application/json"),), b"{bad"),
    )
    for response in invalid_responses:
        invalid_client = ApiClient(
            "https://api.example.test/",
            transport=ScriptedTransport([response]),
        )
        try:
            invalid_client.get_json("data")
        except ResponseValidationError:
            pass
        else:
            raise AssertionError("invalid successful response must fail")

    oversized_client = ApiClient(
        "https://api.example.test/",
        transport=ScriptedTransport(
            [HttpResponse(200, (("Content-Type", "application/json"),), b"1234")]
        ),
        max_response_bytes=3,
    )
    try:
        oversized_client.get_json("large")
    except ResponseValidationError:
        pass
    else:
        raise AssertionError("oversized response must fail")

    pacing_clock = FakeClock(start=10.0)
    limiter = PacedRateLimiter(
        2.0,
        clock=pacing_clock.clock,
        sleeper=pacing_clock.sleep,
    )
    limiter.acquire()
    limiter.acquire()
    assert pacing_clock.sleeps == [0.5]

    safe_headers = redact_headers(
        {"Authorization": "Bearer secret", "Accept": "application/json"}
    )
    assert safe_headers == {
        "Authorization": "[REDACTED]",
        "Accept": "application/json",
    }

    pages = {
        None: (["a", "b"], "next"),
        "next": (["c"], None),
    }
    assert collect_cursor_pages(lambda cursor: pages[cursor]) == ("a", "b", "c")

    repeated = {
        None: (["a"], "same"),
        "same": (["b"], "same"),
    }
    try:
        collect_cursor_pages(lambda cursor: repeated[cursor])
    except RuntimeError:
        pass
    else:
        raise AssertionError("repeated cursor must fail")

    window = parse_rate_limit_headers(
        {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "105",
        },
        now_epoch=100,
    )
    assert window == RateLimitWindow(0, 105.0, 5.0)


def main() -> None:
    """Run a deterministic API-client demonstration."""
    run_self_checks()
    clock = FakeClock()
    transport = ScriptedTransport(
        [
            HttpResponse(
                429,
                (("Retry-After", "1"), ("Content-Type", "application/json")),
                b'{"error":"rate limited"}',
            ),
            _json_response(
                {
                    "day": 25,
                    "lesson": "HTTP clients, retries, and rate limits",
                }
            ),
        ]
    )
    client = ApiClient(
        "https://course.example.test/api/",
        transport=transport,
        sleeper=clock.sleep,
        wall_clock=clock.clock,
    )
    lesson = client.get_json("lessons/25")
    print(f"Result: {lesson}")
    print(f"Attempts: {len(transport.requests)}")
    print(f"Simulated retry waits: {clock.sleeps}")
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
