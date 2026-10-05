"""Day 31: Web and backend service architecture.

Learning goals
--------------
1. Trace an HTTP request through parsing, routing, application logic, and a
   response boundary.
2. Keep transport code separate from domain and service code.
3. Build a small WSGI-compatible JSON API with explicit routes and errors.
4. Validate JSON and query parameters at the edge, then pass trusted values
   inward.
5. Test the whole request/response path deterministically without opening a
   network socket.

Teaching notes
--------------
- A web service is a boundary: bytes arrive with a method, path, headers, and
  body; the service returns a status, headers, and bytes. The boundary should
  translate once, then delegate to application code.
- Routing is a policy. Keep allowed methods and path parameters explicit instead
  of evaluating arbitrary user input or dynamically importing handlers.
- The domain service below knows tasks, IDs, and validation. It does not know
  about WSGI, JSON, status codes, or sockets.
- The WSGI adapter converts an environ mapping into a Request and a Response
  back into the server's start_response callback. A real server can host the
  same callable, but the lesson tests it with an in-memory harness.
- Validate content length, UTF-8, JSON shape, unknown fields, and query values at
  the edge. Return useful 4xx responses without exposing tracebacks.
- A 500 response is deliberately generic. Production systems should log a
  correlation/request ID internally while keeping secrets and implementation
  details out of the client response.
- HTTP handlers should be small orchestration functions. Transactions,
  authentication, rate limits, observability, and durable storage belong at
  explicit boundaries, which makes them replaceable in later lessons.

Run this file with Python 3.10+.
It uses only the standard library and never contacts a network.

Practice exercises
------------------
1. Add PATCH /tasks/<task_id> to mark a task complete. Require a JSON boolean
   field named "done", reject unknown fields, and return the updated task.
2. Add a GET /tasks?limit=N parameter with a safe maximum and a deterministic
   400 response for invalid values.
3. Add request-id middleware that copies a validated X-Request-ID into the
   response and generates one with an injected factory when absent.

Expert challenge: production task service
------------------------------------------
Turn this single-file example into a package with domain, application, and
adapter modules. Replace the in-memory store with the repository port from
Day 30, add authentication and rate limiting at the edge, publish structured
audit events, expose OpenAPI documentation, and run it behind a real WSGI or
ASGI server. Add unit, integration, contract, and load tests. Document timeout,
retry, idempotency, graceful-shutdown, and deployment behavior.

Solution guidance
-----------------
1. Keep Task and TaskService independent from HTTP and JSON. Introduce a
   TaskRepository protocol before adding SQLite or another database.
2. Match routes by fixed segments and validated parameters; never concatenate
   untrusted SQL or shell fragments.
3. Make each response include a content type and byte length. Use 201 plus a
   Location header for creation, 404 for a missing task, 405 with Allow for a
   known path and wrong method, and 400 for malformed input.
4. Test the WSGI boundary with fake environ objects. Assert status, headers,
   body shape, malformed UTF-8, unknown JSON keys, empty titles, missing IDs,
   and repeated requests.
5. Add observability and security as separate middleware so the application
   remains deterministic and easy to test.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
import json
from urllib.parse import parse_qs, urlsplit


class BadRequest(ValueError):
    """The client supplied malformed or invalid request data."""


class RouteNotFound(LookupError):
    """No registered path matches the request."""


class MethodNotAllowed(LookupError):
    """The path exists, but this HTTP method is not registered."""

    def __init__(self, allowed: Iterable[str]) -> None:
        self.allowed = tuple(sorted(set(allowed)))
        super().__init__("method is not allowed")


class NotFound(LookupError):
    """A requested domain object does not exist."""


def _clean_text(value: str, *, field: str, max_length: int = 120) -> str:
    """Normalize bounded text and reject empty or non-text values."""
    if not isinstance(value, str):
        raise BadRequest(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise BadRequest(f"{field} cannot be empty")
    if len(cleaned) > max_length:
        raise BadRequest(f"{field} is too long")
    return cleaned


@dataclass(frozen=True, slots=True)
class Request:
    """Transport-neutral request data produced by the WSGI adapter."""

    method: str
    path: str
    query: Mapping[str, tuple[str, ...]]
    headers: Mapping[str, str]
    body: bytes

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, object],
        *,
        max_body_bytes: int = 16_384,
    ) -> Request:
        """Parse a WSGI environ while enforcing small, explicit limits."""
        method = str(environ.get("REQUEST_METHOD", "GET")).upper().strip()
        if not method or any(not ("A" <= char <= "Z") for char in method):
            raise BadRequest("invalid HTTP method")

        raw_path = str(environ.get("PATH_INFO", "/"))
        if not raw_path.startswith("/") or "\x00" in raw_path:
            raise BadRequest("invalid path")
        path = "/" if raw_path == "/" else raw_path.rstrip("/") or "/"

        raw_query = str(environ.get("QUERY_STRING", ""))
        query_values = parse_qs(raw_query, keep_blank_values=True)
        query = {key: tuple(values) for key, values in query_values.items()}

        headers: dict[str, str] = {}
        for key, value in environ.items():
            if not isinstance(key, str):
                continue
            if key.startswith("HTTP_"):
                header_name = key[5:].replace("_", "-").casefold()
                headers[header_name] = str(value)
        for key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            value = environ.get(key)
            if value is not None:
                headers[key.replace("_", "-").casefold()] = str(value)

        raw_length = headers.get("content-length", "0").strip()
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise BadRequest("content-length must be an integer") from exc
        if length < 0 or length > max_body_bytes:
            raise BadRequest("request body is outside the permitted size")
        stream = environ.get("wsgi.input")
        if length and not hasattr(stream, "read"):
            raise BadRequest("request body stream is missing")
        body = b"" if not length else stream.read(length)  # type: ignore[union-attr]
        if not isinstance(body, bytes) or len(body) != length:
            raise BadRequest("request body was truncated")
        return cls(method, path, query, headers, body)


_STATUS_PHRASES = {
    200: "OK",
    201: "Created",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    500: "Internal Server Error",
}


@dataclass(frozen=True, slots=True)
class Response:
    """A complete HTTP response before it is translated to WSGI."""

    status: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()

    @classmethod
    def json(
        cls,
        payload: object,
        *,
        status: int = 200,
        headers: Iterable[tuple[str, str]] = (),
    ) -> Response:
        """Encode one JSON response with deterministic formatting."""
        if status not in _STATUS_PHRASES:
            raise ValueError(f"unsupported response status: {status}")
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        base_headers = [("Content-Type", "application/json; charset=utf-8")]
        base_headers.extend(headers)
        return cls(status, body, tuple(base_headers))

    def to_wsgi(self, start_response: Callable[[str, list[tuple[str, str]]], object]):
        """Send status and headers to a WSGI server and return body chunks."""
        phrase = _STATUS_PHRASES[self.status]
        headers = list(self.headers)
        headers.append(("Content-Length", str(len(self.body))))
        start_response(f"{self.status} {phrase}", headers)
        return [self.body]


RouteHandler = Callable[[Request, Mapping[str, str]], Response]


class Router:
    """Small explicit router supporting fixed and <parameter> segments."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, str, RouteHandler]] = []

    def add(self, method: str, pattern: str, handler: RouteHandler) -> None:
        """Register one uppercase method and a slash-normalized pattern."""
        clean_method = method.upper().strip()
        if not clean_method or not pattern.startswith("/"):
            raise ValueError("routes require a method and absolute pattern")
        if not callable(handler):
            raise TypeError("route handler must be callable")
        normalized = "/" if pattern == "/" else pattern.rstrip("/")
        self._routes.append((clean_method, normalized, handler))

    @staticmethod
    def _match(pattern: str, path: str) -> dict[str, str] | None:
        pattern_parts = [] if pattern == "/" else pattern.strip("/").split("/")
        path_parts = [] if path == "/" else path.strip("/").split("/")
        if len(pattern_parts) != len(path_parts):
            return None
        values: dict[str, str] = {}
        for expected, actual in zip(pattern_parts, path_parts):
            if expected.startswith("<") and expected.endswith(">"):
                name = expected[1:-1]
                if not name or "/" in actual or not actual:
                    return None
                values[name] = actual
            elif expected != actual:
                return None
        return values

    def dispatch(self, request: Request) -> Response:
        """Find a handler or raise a precise routing error."""
        allowed: list[str] = []
        for method, pattern, handler in self._routes:
            values = self._match(pattern, request.path)
            if values is None:
                continue
            allowed.append(method)
            if method == request.method:
                return handler(request, values)
        if allowed:
            raise MethodNotAllowed(allowed)
        raise RouteNotFound(request.path)


@dataclass(frozen=True, slots=True)
class Task:
    """Domain value that contains no HTTP or JSON concerns."""

    task_id: str
    title: str
    done: bool
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "done": self.done,
            "id": self.task_id,
            "title": self.title,
        }


class TaskService:
    """Application service for task rules and in-memory persistence."""

    def __init__(
        self,
        *,
        id_factory: Callable[[], str],
        clock: Callable[[], str],
    ) -> None:
        if not callable(id_factory) or not callable(clock):
            raise TypeError("id_factory and clock must be callable")
        self._id_factory = id_factory
        self._clock = clock
        self._tasks: dict[str, Task] = {}

    def create(self, title: str) -> Task:
        """Validate and store one new task."""
        clean_title = _clean_text(title, field="title")
        task_id = _clean_text(self._id_factory(), field="task_id", max_length=80)
        if task_id in self._tasks:
            raise ValueError("task ID factory returned a duplicate")
        task = Task(task_id, clean_title, False, self._clock())
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Task:
        """Return one task or a domain-level not-found error."""
        clean_id = _clean_text(task_id, field="task_id", max_length=80)
        try:
            return self._tasks[clean_id]
        except KeyError as exc:
            raise NotFound("task does not exist") from exc

    def list(self, *, done: bool | None = None) -> tuple[Task, ...]:
        """Return tasks in insertion order, optionally filtered by completion."""
        if done is None:
            return tuple(self._tasks.values())
        return tuple(task for task in self._tasks.values() if task.done is done)


class ApiApplication:
    """Application boundary that maps trusted requests to JSON responses."""

    def __init__(self, service: TaskService) -> None:
        if not isinstance(service, TaskService):
            raise TypeError("service must be a TaskService")
        self._service = service
        self._router = Router()
        self._router.add("GET", "/health", self._health)
        self._router.add("GET", "/tasks", self._list_tasks)
        self._router.add("POST", "/tasks", self._create_task)
        self._router.add("GET", "/tasks/<task_id>", self._get_task)

    @staticmethod
    def _health(request: Request, params: Mapping[str, str]) -> Response:
        del request, params
        return Response.json({"service": "task-api", "status": "ok"})

    @staticmethod
    def _json_object(request: Request) -> dict[str, object]:
        """Decode an object and reject malformed or unexpected JSON."""
        if not request.body:
            raise BadRequest("JSON body is required")
        try:
            decoded = json.loads(request.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BadRequest("body must be valid UTF-8 JSON") from exc
        if not isinstance(decoded, dict):
            raise BadRequest("JSON body must be an object")
        return decoded

    def _create_task(self, request: Request, params: Mapping[str, str]) -> Response:
        del params
        payload = self._json_object(request)
        if set(payload) != {"title"}:
            raise BadRequest("body must contain only the title field")
        task = self._service.create(payload["title"])  # type: ignore[arg-type]
        return Response.json(
            task.as_dict(),
            status=201,
            headers=(("Location", f"/tasks/{task.task_id}"),),
        )

    def _list_tasks(self, request: Request, params: Mapping[str, str]) -> Response:
        del params
        values = request.query.get("done")
        done: bool | None = None
        if values:
            selected = values[-1].casefold()
            if selected not in {"true", "false"}:
                raise BadRequest("done must be true or false")
            done = selected == "true"
        return Response.json({"tasks": [task.as_dict() for task in self._service.list(done=done)]})

    def _get_task(self, request: Request, params: Mapping[str, str]) -> Response:
        del request
        task = self._service.get(params["task_id"])
        return Response.json(task.as_dict())

    def handle(self, environ: Mapping[str, object]) -> Response:
        """Handle one request and convert known failures to stable 4xx JSON."""
        try:
            request = Request.from_environ(environ)
            return self._router.dispatch(request)
        except BadRequest as exc:
            return Response.json({"error": str(exc)}, status=400)
        except MethodNotAllowed as exc:
            return Response.json(
                {"error": "method not allowed"},
                status=405,
                headers=(("Allow", ", ".join(exc.allowed)),),
            )
        except (RouteNotFound, NotFound):
            return Response.json({"error": "resource not found"}, status=404)
        except Exception:
            # Do not expose stack traces or implementation details to callers.
            return Response.json({"error": "internal server error"}, status=500)


def make_wsgi_application(service: TaskService) -> Callable[..., list[bytes]]:
    """Adapt the application boundary to the WSGI callable contract."""
    app = ApiApplication(service)

    def wsgi_app(
        environ: Mapping[str, object],
        start_response: Callable[[str, list[tuple[str, str]]], object],
    ) -> list[bytes]:
        return app.handle(environ).to_wsgi(start_response)

    return wsgi_app


def invoke(
    app: Callable[..., list[bytes]],
    method: str,
    target: str,
    *,
    body: bytes = b"",
    headers: Mapping[str, str] = {},
) -> tuple[str, dict[str, str], bytes]:
    """Exercise a WSGI app without binding a port or making network calls."""
    parsed = urlsplit(target)
    environ: dict[str, object] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": parsed.path or "/",
        "QUERY_STRING": parsed.query,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": BytesIO(body),
    }
    for name, value in headers.items():
        environ_key = "HTTP_" + name.upper().replace("-", "_")
        environ[environ_key] = value
    captured: dict[str, object] = {}

    def start_response(status: str, response_headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = response_headers

    chunks = app(environ, start_response)
    status = captured.get("status")
    raw_headers = captured.get("headers")
    if not isinstance(status, str) or not isinstance(raw_headers, list):
        raise AssertionError("WSGI app did not call start_response correctly")
    return status, dict(raw_headers), b"".join(chunks)


def run_self_checks() -> None:
    """Check routing, validation, status codes, headers, and response bodies."""
    ids = iter(("task-001", "task-002"))
    service = TaskService(
        id_factory=lambda: next(ids),
        clock=lambda: "2026-10-05T00:00:00+00:00",
    )
    app = make_wsgi_application(service)

    status, headers, body = invoke(app, "GET", "/health")
    assert status == "200 OK"
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert int(headers["Content-Length"]) == len(body)
    assert json.loads(body) == {"service": "task-api", "status": "ok"}

    status, _, body = invoke(app, "POST", "/tasks", body=b"{}")
    assert status == "400 Bad Request"
    assert json.loads(body)["error"] == "body must contain only the title field"

    status, _, body = invoke(app, "POST", "/tasks", body=b'{"title":"  Write docs  "}')
    assert status == "201 Created"
    created = json.loads(body)
    assert created["id"] == "task-001"
    assert created["title"] == "Write docs"

    status, headers, body = invoke(app, "GET", "/tasks")
    assert status == "200 OK"
    assert headers["Content-Length"] == str(len(body))
    assert json.loads(body)["tasks"][0]["id"] == "task-001"

    status, _, body = invoke(app, "GET", "/tasks/task-001")
    assert status == "200 OK"
    assert json.loads(body)["done"] is False

    status, _, body = invoke(app, "GET", "/tasks?done=maybe")
    assert status == "400 Bad Request"
    assert "true or false" in json.loads(body)["error"]

    status, headers, body = invoke(app, "PUT", "/tasks")
    assert status == "405 Method Not Allowed"
    assert headers["Allow"] == "GET, POST"
    assert json.loads(body)["error"] == "method not allowed"

    status, _, body = invoke(app, "GET", "/tasks/missing")
    assert status == "404 Not Found"
    assert json.loads(body) == {"error": "resource not found"}

    status, _, body = invoke(app, "POST", "/tasks", body=b"\xff")
    assert status == "400 Bad Request"
    assert "valid UTF-8 JSON" in json.loads(body)["error"]

    status, _, body = invoke(app, "POST", "/tasks", body=b'["not", "an", "object"]')
    assert status == "400 Bad Request"
    assert json.loads(body)["error"] == "JSON body must be an object"

    status, _, body = invoke(app, "POST", "/tasks", body=b'{"title":""}')
    assert status == "400 Bad Request"
    assert json.loads(body)["error"] == "title cannot be empty"

    status, _, body = invoke(app, "GET", "/tasks/")
    assert status == "200 OK"
    assert len(json.loads(body)["tasks"]) == 1


def main() -> None:
    """Run a deterministic API demonstration."""
    run_self_checks()
    service = TaskService(
        id_factory=iter(("demo-001",)).__next__,
        clock=lambda: "2026-10-05T00:00:00+00:00",
    )
    app = make_wsgi_application(service)
    status, headers, body = invoke(
        app,
        "POST",
        "/tasks",
        body=json.dumps({"title": "Study web boundaries"}).encode("utf-8"),
    )
    print("Status:", status)
    print("Location:", headers["Location"])
    print("Body:", body.decode("utf-8"))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
