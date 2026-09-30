"""Day 26: HTTP security, authentication, and secure API boundaries.

Learning goals
--------------
1. Keep secrets out of source code, URLs, logs, tracebacks, and reprs.
2. Build an HTTPS-only API boundary that rejects URL and host escapes.
3. Model API-key and bearer authentication explicitly.
4. Verify timestamped HMAC webhooks with constant-time comparison.
5. Test security decisions deterministically without a real network.

Teaching notes
--------------
- Authentication answers who is calling; authorization answers what it may do.
  Use the smallest scopes possible, rotate credentials, and prefer HTTPS.
- Bearer tokens are reusable by anyone who obtains them. Put them in headers,
  never query strings, and load them from a secret manager or environment at the
  process boundary.
- URL construction is a security boundary. Reject absolute endpoints, schemes,
  host overrides, backslashes, path traversal, unexpected hosts, and unsafe
  redirects. An HTTPS URL can still redirect to an internal service.
- HMAC signatures bind a body to a shared secret. Sign a documented canonical
  value containing timestamp and nonce, reject stale timestamps, and use
  hmac.compare_digest rather than ==.
- Bound body and header sizes before expensive work. Authenticate a webhook before
  parsing untrusted JSON, then enqueue it behind an idempotency key.
- Logs should contain safe request IDs, host, status, latency, and reason category;
  never credentials or full untrusted response bodies. Redaction is defense in
  depth, not a reason to log secrets.

Run this file with Python 3.10+. It uses only the standard library. The runnable
example prepares local request metadata and verifies a local webhook; no network
is used.

Practice exercises
------------------
1. Implement BasicAuth that rejects newlines and encodes credentials with base64.
2. Extend redact_headers for custom secret headers without mutating the input.
3. Add a scope checker that rejects permissions outside an allow-list and returns
   a sorted immutable result.

Solutions appear below the boundary implementation.

Expert challenge: secure webhook ingestion service
---------------------------------------------------
Design a multi-tenant billing webhook endpoint. Verify a timestamped HMAC,
reject replayed event IDs, validate a versioned payload, and enqueue work
transactionally. Add key rotation, per-tenant rate limits, structured audit
events, idempotent delivery, and tests for every rejection path.

Solution guidance
-----------------
1. Inject transport, clock, replay storage, schema validation, and queue adapters.
2. Compute a documented canonical byte string and compare MACs with compare_digest.
3. Store (tenant, event_id) and the enqueue record in one transaction.
4. Bound body, timestamp, queue, and pagination limits before parsing.
5. Try a current and bounded previous key; expose key IDs only in internal metrics.
6. Test valid, malformed, stale, duplicate, oversized, schema-drift, isolated,
   queue-failure, and rotated-key deliveries.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from typing import Protocol
from urllib.parse import urljoin, urlsplit


class SecurityBoundaryError(ValueError):
    """Expected rejection at an authentication or network boundary."""


def _clean_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    result = value.strip()
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return result


def _positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _https_origin(value: str, *, field: str) -> tuple[str, str]:
    cleaned = _clean_text(value, field=field)
    parts = urlsplit(cleaned)
    if parts.scheme.casefold() != "https":
        raise SecurityBoundaryError(f"{field} must use HTTPS")
    if not parts.hostname or parts.username or parts.password:
        raise SecurityBoundaryError(f"{field} must have a host without credentials")
    if parts.query or parts.fragment:
        raise SecurityBoundaryError(f"{field} cannot contain a query or fragment")
    path = parts.path if parts.path.endswith("/") else parts.path + "/"
    return parts.hostname.casefold(), path


class AuthProvider(Protocol):
    def apply(self, headers: dict[str, str]) -> None:
        """Add authentication to a header mapping without printing the secret."""


@dataclass(frozen=True, slots=True, repr=False)
class BearerAuth:
    """A token is deliberately hidden from repr and sent only as a header."""

    token: str

    def __post_init__(self) -> None:
        token = _clean_text(self.token, field="token")
        if any(char.isspace() for char in token):
            raise SecurityBoundaryError("token cannot contain whitespace")
        object.__setattr__(self, "token", token)

    def __repr__(self) -> str:
        return "BearerAuth(token='[REDACTED]')"

    def apply(self, headers: dict[str, str]) -> None:
        headers["Authorization"] = f"Bearer {self.token}"


@dataclass(frozen=True, slots=True, repr=False)
class ApiKeyAuth:
    """An API key belongs in a header, never in a query parameter."""

    key: str
    header_name: str = "X-API-Key"

    def __post_init__(self) -> None:
        key = _clean_text(self.key, field="key")
        header = _clean_text(self.header_name, field="header_name")
        if any(char.isspace() for char in key):
            raise SecurityBoundaryError("key cannot contain whitespace")
        if any(char in header for char in "\r\n:"):
            raise SecurityBoundaryError("header_name contains invalid characters")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "header_name", header)

    def __repr__(self) -> str:
        return "ApiKeyAuth(key='[REDACTED]')"

    def apply(self, headers: dict[str, str]) -> None:
        headers[self.header_name] = self.key


def basic_auth_header(username: str, password: str) -> str:
    """Practice solution: encode Basic credentials without exposing them."""
    user = _clean_text(username, field="username")
    secret = _clean_text(password, field="password")
    if any(char in user + secret for char in "\r\n"):
        raise SecurityBoundaryError("Basic credentials cannot contain newlines")
    encoded = base64.b64encode(f"{user}:{secret}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a safe copy for structured logs."""
    exact = {
        "authorization", "cookie", "proxy-authorization", "set-cookie",
        "x-api-key", "x-signature",
    }
    redacted: dict[str, str] = {}
    for name, value in headers.items():
        clean_name = _clean_text(name, field="header name")
        if not isinstance(value, str):
            raise TypeError("header values must be text")
        lowered = clean_name.casefold()
        sensitive = (
            lowered in exact
            or "token" in lowered
            or "secret" in lowered
            or lowered.endswith("-key")
        )
        redacted[clean_name] = "[REDACTED]" if sensitive else value
    return redacted


def allowed_scopes(
    requested: Sequence[str], allowed: Sequence[str]
) -> tuple[str, ...]:
    """Practice solution: enforce least privilege and return stable output."""
    if isinstance(requested, (str, bytes)) or isinstance(allowed, (str, bytes)):
        raise TypeError("scopes must be sequences of text")
    permitted = {_clean_text(scope, field="allowed scope") for scope in allowed}
    chosen = {_clean_text(scope, field="requested scope") for scope in requested}
    if chosen - permitted:
        raise SecurityBoundaryError("requested scope is not allowed")
    return tuple(sorted(chosen))


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """A network-independent request snapshot for an HTTP adapter."""

    method: str
    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes | None


class SecureApiBoundary:
    """Prepare HTTPS requests while enforcing a host and path allow-list."""

    _METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})

    def __init__(
        self,
        base_url: str,
        *,
        auth: AuthProvider,
        allowed_hosts: Sequence[str] | None = None,
        max_body_bytes: int = 1_000_000,
    ) -> None:
        host, path = _https_origin(base_url, field="base_url")
        hosts = {host}
        if allowed_hosts is not None:
            if isinstance(allowed_hosts, (str, bytes)):
                raise TypeError("allowed_hosts must be a sequence")
            hosts = {
                _clean_text(item, field="allowed host").casefold()
                for item in allowed_hosts
            }
            if not hosts or host not in hosts:
                raise SecurityBoundaryError("base host must be in allowed_hosts")
        if not callable(getattr(auth, "apply", None)):
            raise TypeError("auth must provide apply(headers)")
        self._base_url = urljoin(f"https://{host}/", path.lstrip("/"))
        self._allowed_hosts = frozenset(hosts)
        self._auth = auth
        self._max_body_bytes = _positive_int(max_body_bytes, field="max_body_bytes")

    def _safe_url(self, endpoint: str) -> str:
        endpoint = _clean_text(endpoint, field="endpoint")
        parts = urlsplit(endpoint)
        unsafe = (
            parts.scheme or parts.netloc or parts.query or parts.fragment
            or endpoint.startswith(("/", "\\"))
            or "\\" in endpoint
            or any(part == ".." for part in parts.path.split("/"))
            or any(char.isspace() for char in endpoint)
        )
        if unsafe:
            raise SecurityBoundaryError("endpoint must be a safe relative path")
        url = urljoin(self._base_url, endpoint)
        parsed = urlsplit(url)
        if parsed.scheme.casefold() != "https" or parsed.hostname not in self._allowed_hosts:
            raise SecurityBoundaryError("resolved endpoint is outside the allow-list")
        return url

    @staticmethod
    def _json_bytes(value: object) -> bytes:
        try:
            return json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise SecurityBoundaryError("body is not finite JSON") from error

    def prepare(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: object | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> PreparedRequest:
        method = _clean_text(method, field="method").upper()
        if method not in self._METHODS:
            raise SecurityBoundaryError("unsupported HTTP method")
        result: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "Python-Courses-Day-026/1.0",
        }
        if headers is not None:
            for name, value in headers.items():
                clean_name = _clean_text(name, field="header name")
                if any(char in clean_name for char in "\r\n:"):
                    raise SecurityBoundaryError("invalid header name")
                if not isinstance(value, str) or any(char in value for char in "\r\n"):
                    raise SecurityBoundaryError("invalid header value")
                result[clean_name] = value
        body = None if json_body is None else self._json_bytes(json_body)
        if body is not None:
            if len(body) > self._max_body_bytes:
                raise SecurityBoundaryError("request body is too large")
            result["Content-Type"] = "application/json; charset=utf-8"
        self._auth.apply(result)
        return PreparedRequest(method, self._safe_url(endpoint), tuple(sorted(result.items())), body)


def canonical_signature_input(body: bytes, *, timestamp: int, nonce: str) -> bytes:
    """A documented canonical form for HMAC signing."""
    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise TypeError("timestamp must be an integer")
    clean_nonce = _clean_text(nonce, field="nonce")
    if any(char.isspace() for char in clean_nonce):
        raise SecurityBoundaryError("nonce cannot contain whitespace")
    return b".".join((str(timestamp).encode(), clean_nonce.encode(), body))


def sign_webhook(body: bytes, *, secret: str, timestamp: int, nonce: str) -> str:
    key = _clean_text(secret, field="secret").encode()
    message = canonical_signature_input(body, timestamp=timestamp, nonce=nonce)
    return "v1=" + hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_webhook_signature(
    body: bytes,
    *,
    signature: str,
    secret: str,
    timestamp: int,
    nonce: str,
    now: int,
    tolerance_seconds: int = 300,
) -> None:
    """Raise one generic error for stale and invalid signatures."""
    tolerance = _positive_int(tolerance_seconds, field="tolerance_seconds")
    if isinstance(now, bool) or not isinstance(now, int):
        raise TypeError("now must be an integer")
    if abs(now - timestamp) > tolerance:
        raise SecurityBoundaryError("webhook authentication failed")
    expected = sign_webhook(body, secret=secret, timestamp=timestamp, nonce=nonce)
    provided = _clean_text(signature, field="signature")
    if not hmac.compare_digest(expected, provided):
        raise SecurityBoundaryError("webhook authentication failed")


def parse_secret_from_environment(
    name: str, *, environ: Mapping[str, str] | None = None
) -> str:
    """Read a secret once at the process boundary."""
    variable = _clean_text(name, field="environment variable")
    source = os.environ if environ is None else environ
    try:
        return _clean_text(source[variable], field="secret")
    except KeyError as error:
        raise SecurityBoundaryError("required secret is missing") from error


class WebhookHandler(Protocol):
    def accept(self, event_id: str, payload: Mapping[str, object]) -> bool:
        """Return False when the event was already accepted."""


def accept_authenticated_event(
    event_id: str,
    payload: Mapping[str, object],
    *,
    handler: WebhookHandler,
) -> bool:
    """Validate a small event envelope before application code runs."""
    clean_id = _clean_text(event_id, field="event_id")
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Mapping):
        raise SecurityBoundaryError("payload must be an object")
    if not clean_id.isascii() or len(clean_id) > 128:
        raise SecurityBoundaryError("event_id has an invalid shape")
    if not isinstance(payload.get("type"), str) or not payload["type"].strip():
        raise SecurityBoundaryError("payload type is required")
    if not callable(getattr(handler, "accept", None)):
        raise TypeError("handler must provide accept(event_id, payload)")
    return bool(handler.accept(clean_id, payload))


@dataclass
class RecordingHandler:
    accepted: dict[str, Mapping[str, object]]

    def accept(self, event_id: str, payload: Mapping[str, object]) -> bool:
        if event_id in self.accepted:
            return False
        self.accepted[event_id] = dict(payload)
        return True


def run_self_checks() -> None:
    """Exercise success paths and important edge cases."""
    auth = BearerAuth("token-123")
    assert repr(auth) == "BearerAuth(token='[REDACTED]')"
    boundary = SecureApiBoundary(
        "https://api.example.test/v1",
        auth=auth,
        allowed_hosts=("api.example.test",),
        max_body_bytes=100,
    )
    request = boundary.prepare(
        "POST", "events/today", json_body={"ok": True}, headers={"X-Request-ID": "demo-1"}
    )
    assert request.url == "https://api.example.test/v1/events/today"
    assert dict(request.headers)["Authorization"] == "Bearer token-123"
    assert request.body == b'{"ok":true}'
    for bad_endpoint in (
        "https://internal.example.test/admin",
        "//internal.example.test/admin",
        "/absolute",
        "../escape",
        r"nested\\escape",
        "items?token=secret",
    ):
        try:
            boundary.prepare("GET", bad_endpoint)
        except SecurityBoundaryError:
            pass
        else:
            raise AssertionError("unsafe endpoint was accepted")

    assert basic_auth_header("alice", "p@ss") == "Basic YWxpY2U6cEBzcw=="
    assert redact_headers(
        {"Authorization": "Bearer secret", "X-Trace": "ok", "X-Api-Key": "secret"}
    ) == {"Authorization": "[REDACTED]", "X-Trace": "ok", "X-Api-Key": "[REDACTED]"}
    assert allowed_scopes(("read:invoices", "read:invoices"), ("read:invoices",)) == (
        "read:invoices",
    )
    try:
        allowed_scopes(("admin",), ("read:invoices",))
    except SecurityBoundaryError:
        pass
    else:
        raise AssertionError("unknown scope was accepted")

    body = b'{"event":"paid"}'
    signature = sign_webhook(body, secret="shared-secret", timestamp=100, nonce="n-1")
    verify_webhook_signature(
        body, signature=signature, secret="shared-secret",
        timestamp=100, nonce="n-1", now=120,
    )
    for bad_signature, bad_timestamp in (("v1=bad", 100), (signature, 1_000)):
        try:
            verify_webhook_signature(
                body, signature=bad_signature, secret="shared-secret",
                timestamp=bad_timestamp, nonce="n-1", now=120,
            )
        except SecurityBoundaryError:
            pass
        else:
            raise AssertionError("invalid webhook was accepted")

    handler = RecordingHandler({})
    payload = {"type": "invoice.paid", "amount": 25}
    assert accept_authenticated_event("evt-1", payload, handler=handler)
    assert not accept_authenticated_event("evt-1", payload, handler=handler)
    assert parse_secret_from_environment(
        "COURSE_SECRET", environ={"COURSE_SECRET": "local-only"}
    ) == "local-only"


def main() -> None:
    """Run a deterministic, network-free demonstration."""
    run_self_checks()
    boundary = SecureApiBoundary(
        "https://billing.example.test/api",
        auth=ApiKeyAuth("demo-key"),
        allowed_hosts=("billing.example.test",),
    )
    request = boundary.prepare("GET", "invoices", headers={"X-Request-ID": "demo-26"})
    print(f"Prepared URL: {request.url}")
    print(f"Safe headers: {redact_headers(dict(request.headers))}")
    body = b'{"event":"paid"}'
    signature = sign_webhook(body, secret="demo-secret", timestamp=1_000, nonce="demo-1")
    verify_webhook_signature(
        body, signature=signature, secret="demo-secret",
        timestamp=1_000, nonce="demo-1", now=1_050,
    )
    print("Webhook signature verified.")
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
