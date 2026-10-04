"""Day 30: Architecture and design patterns.

Learning goals
--------------
1. Separate domain rules, application orchestration, and side-effect adapters.
2. Use Strategy, Factory, Repository, Command, and Adapter patterns when they
   clarify a real change point rather than adding ceremony.
3. Inject policies, storage, identifiers, and event sinks for deterministic tests.
4. Make idempotent commands safe to retry and detect conflicting reuse.
5. Keep dependencies pointing inward so the domain does not know about SQL,
   HTTP clients, or terminal output.

Teaching notes
--------------
- Architecture is a set of boundaries and dependency directions. A useful
  small system can have a domain layer (values and rules), an application layer
  (use cases), and adapter layers (storage, transport, and presentation).
- The Strategy pattern represents a replaceable policy. Here, pricing policies
  can change without changing OrderService.
- A Factory centralizes a safe choice among known implementations. A registry
  or allow-list is preferable to importing a class named by untrusted input.
- A Repository is a persistence port. The service asks for orders and saves
  receipts; an adapter decides whether the data lives in memory, SQLite, or an
  API.
- A Command is a validated request object. It makes a use case explicit and
  gives a queue, retry layer, or audit log a stable value to carry.
- An Adapter translates one interface to another. CallbackEventSink lets an
  existing callable satisfy EventSink without making the service know about it.
- Dependency injection means collaborators arrive at a boundary. It avoids
  hidden globals and lets tests replace a clock, ID source, database, or network
  client with a deterministic fake.
- Idempotency is an application contract, not merely a dictionary lookup. A
  repeated key with the same request returns the original result; the same key
  for different input is a conflict. Durable storage must enforce this rule
  atomically in production.
- Keep transactions and failure policy at the boundary that owns the side
  effect. If saving succeeds but event publication fails, decide whether to
  retry, use an outbox, or report a partial result. Do not pretend the two
  effects are one transaction unless they really are.
- Patterns are vocabulary, not goals. Prefer a plain function when no
  variation or boundary exists. Measure the operational behavior of a design
  with the profiling ideas from Day 29.

Run this file with Python 3.10+ to execute deterministic examples and checks.
It uses only the standard library and never contacts a payment service.

Practice exercises
------------------
1. Add a FreeShippingPolicy strategy that changes only the quote calculation.
2. Implement a SQLiteOrderStore with parameterized SQL and a unique
   idempotency-key constraint while keeping OrderService unchanged.
3. Implement a JsonEventSink adapter that serializes OrderPlaced events without
   including the request fingerprint.

Solutions appear below the main example.

Expert challenge: production order workflow
-------------------------------------------
Evolve the in-memory example into a package with domain, application, and
adapter modules. Add a durable repository, an outbox table, a retry worker,
structured audit events, and a CLI/API boundary. Prove that a crash before
commit cannot charge twice, that conflicting idempotency keys are rejected, and
that every public boundary has unit, integration, and contract tests.

Solution guidance
-----------------
1. Keep OrderLine, OrderDraft, Quote, and Receipt free of I/O.
2. Make the repository save receipt and idempotency fingerprint in one
   transaction; use a unique key in the database.
3. Publish an outbox record in that transaction, then let a separate worker
   deliver events with bounded retries and a dead-letter policy.
4. Use a stable command ID and injected clock/ID factories for reproducible
   tests. Never put secrets or full card data in events.
5. Test a repeated command, a conflicting command, duplicate storage, event
   failure, malformed input, and graceful shutdown.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import json
from typing import Protocol


class DomainError(ValueError):
    """Base class for expected business-rule failures."""


class ConflictError(DomainError):
    """An idempotency key was reused for different input."""


def _clean_text(value: str, *, field: str) -> str:
    """Return normalized, non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field} cannot be empty")
    return cleaned


def _non_negative_int(value: int, *, field: str) -> int:
    """Accept an integer zero or greater, but reject bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    return value


def _positive_int(value: int, *, field: str) -> int:
    """Accept a positive integer, but reject bool."""
    value = _non_negative_int(value, field=field)
    if value == 0:
        raise ValueError(f"{field} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class OrderLine:
    """Immutable domain data for one product line, in integer cents."""

    name: str
    unit_price_cents: int
    quantity: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, field="name"))
        object.__setattr__(
            self,
            "unit_price_cents",
            _non_negative_int(self.unit_price_cents, field="unit_price_cents"),
        )
        object.__setattr__(
            self,
            "quantity",
            _positive_int(self.quantity, field="quantity"),
        )

    @property
    def subtotal_cents(self) -> int:
        """Return this line's extended price."""
        return self.unit_price_cents * self.quantity


@dataclass(frozen=True, slots=True)
class OrderDraft:
    """Validated input to the place-order use case."""

    lines: tuple[OrderLine, ...]

    def __post_init__(self) -> None:
        lines = tuple(self.lines)
        if not lines:
            raise ValueError("an order must contain at least one line")
        if any(not isinstance(line, OrderLine) for line in lines):
            raise TypeError("lines must contain OrderLine values")
        object.__setattr__(self, "lines", lines)

    @property
    def subtotal_cents(self) -> int:
        return sum(line.subtotal_cents for line in self.lines)

    @property
    def fingerprint(self) -> str:
        """Return a stable, non-secret identity for this exact request."""
        canonical = "|".join(
            f"{line.name}\x1f{line.unit_price_cents}\x1f{line.quantity}"
            for line in self.lines
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Quote:
    """A domain result independent of storage or presentation."""

    subtotal_cents: int
    discount_cents: int
    total_cents: int
    policy_name: str

    def __post_init__(self) -> None:
        subtotal = _non_negative_int(self.subtotal_cents, field="subtotal_cents")
        discount = _non_negative_int(self.discount_cents, field="discount_cents")
        total = _non_negative_int(self.total_cents, field="total_cents")
        if discount > subtotal or total != subtotal - discount:
            raise ValueError("quote totals are inconsistent")
        object.__setattr__(self, "subtotal_cents", subtotal)
        object.__setattr__(self, "discount_cents", discount)
        object.__setattr__(self, "total_cents", total)
        object.__setattr__(
            self,
            "policy_name",
            _clean_text(self.policy_name, field="policy_name"),
        )


class PricingPolicy(Protocol):
    """Strategy port for one pricing rule."""

    @property
    def name(self) -> str:
        """Return a stable policy label."""

    def quote(self, draft: OrderDraft) -> Quote:
        """Price a validated draft."""


class NoDiscountPolicy:
    """Strategy that charges the subtotal."""

    @property
    def name(self) -> str:
        return "standard"

    def quote(self, draft: OrderDraft) -> Quote:
        if not isinstance(draft, OrderDraft):
            raise TypeError("draft must be an OrderDraft")
        return Quote(draft.subtotal_cents, 0, draft.subtotal_cents, self.name)


class PercentDiscountPolicy:
    """Strategy that applies a validated whole-number percentage."""

    def __init__(self, percent: int) -> None:
        if isinstance(percent, bool) or not isinstance(percent, int):
            raise TypeError("percent must be an integer")
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        self._percent = percent

    @property
    def name(self) -> str:
        return f"discount-{self._percent}%"

    def quote(self, draft: OrderDraft) -> Quote:
        if not isinstance(draft, OrderDraft):
            raise TypeError("draft must be an OrderDraft")
        # Integer arithmetic makes the boundary rule explicit and avoids
        # binary floating-point surprises in money calculations.
        discount = (draft.subtotal_cents * self._percent + 50) // 100
        return Quote(
            draft.subtotal_cents,
            discount,
            draft.subtotal_cents - discount,
            self.name,
        )


def build_pricing_policy(name: str, *, percent: int = 0) -> PricingPolicy:
    """Factory: create only an explicitly supported strategy."""
    selected = _clean_text(name, field="pricing policy").casefold()
    if selected == "standard":
        return NoDiscountPolicy()
    if selected == "sale":
        return PercentDiscountPolicy(percent)
    raise ValueError(f"unknown pricing policy: {name!r}")


@dataclass(frozen=True, slots=True)
class Receipt:
    """Idempotent result returned by the application layer."""

    order_id: str
    idempotency_key: str
    fingerprint: str
    quote: Quote

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "order_id", _clean_text(self.order_id, field="order_id")
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _clean_text(self.idempotency_key, field="idempotency_key"),
        )
        if len(self.fingerprint) != 64:
            raise ValueError("fingerprint must be a SHA-256 hex digest")
        object.__setattr__(self, "fingerprint", self.fingerprint.casefold())
        if not isinstance(self.quote, Quote):
            raise TypeError("quote must be a Quote")


class OrderStore(Protocol):
    """Repository port owned by the application boundary."""

    def find_by_idempotency_key(self, key: str) -> Receipt | None:
        """Return a prior receipt, if one exists."""

    def save(self, receipt: Receipt) -> None:
        """Persist a receipt or raise ConflictError."""


class MemoryOrderStore:
    """Repository adapter with deterministic behavior for this lesson."""

    def __init__(self) -> None:
        self._receipts: dict[str, Receipt] = {}

    def find_by_idempotency_key(self, key: str) -> Receipt | None:
        return self._receipts.get(key)

    def save(self, receipt: Receipt) -> None:
        if not isinstance(receipt, Receipt):
            raise TypeError("receipt must be a Receipt")
        existing = self._receipts.get(receipt.idempotency_key)
        if existing is not None:
            if existing.fingerprint != receipt.fingerprint:
                raise ConflictError("idempotency key already belongs to another request")
            return
        self._receipts[receipt.idempotency_key] = receipt

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        """Expose an immutable snapshot for inspection and tests."""
        return tuple(self._receipts.values())


@dataclass(frozen=True, slots=True)
class OrderPlaced:
    """Safe event data; it intentionally omits the request fingerprint."""

    order_id: str
    total_cents: int
    policy_name: str

    def as_json(self) -> str:
        return json.dumps(
            {
                "event": "order.placed",
                "order_id": self.order_id,
                "policy": self.policy_name,
                "total_cents": self.total_cents,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class EventSink(Protocol):
    """Port for durable or external event publication."""

    def publish(self, event: OrderPlaced) -> None:
        """Publish one safe event."""


class MemoryEventSink:
    """Event adapter that stores immutable snapshots."""

    def __init__(self) -> None:
        self._events: list[OrderPlaced] = []

    def publish(self, event: OrderPlaced) -> None:
        if not isinstance(event, OrderPlaced):
            raise TypeError("event must be an OrderPlaced value")
        self._events.append(event)

    @property
    def events(self) -> tuple[OrderPlaced, ...]:
        return tuple(self._events)


class CallbackEventSink:
    """Adapter: turn a callback into the EventSink port."""

    def __init__(self, callback: Callable[[OrderPlaced], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callback = callback

    def publish(self, event: OrderPlaced) -> None:
        if not isinstance(event, OrderPlaced):
            raise TypeError("event must be an OrderPlaced value")
        self._callback(event)


class OrderService:
    """Application use case that coordinates injected ports and strategies."""

    def __init__(
        self,
        pricing: PricingPolicy,
        store: OrderStore,
        events: EventSink,
        *,
        id_factory: Callable[[], str],
    ) -> None:
        if not callable(getattr(pricing, "quote", None)):
            raise TypeError("pricing must provide quote()")
        if not callable(getattr(store, "save", None)) or not callable(
            getattr(store, "find_by_idempotency_key", None)
        ):
            raise TypeError("store must provide find and save")
        if not callable(getattr(events, "publish", None)):
            raise TypeError("events must provide publish()")
        if not callable(id_factory):
            raise TypeError("id_factory must be callable")
        self._pricing = pricing
        self._store = store
        self._events = events
        self._id_factory = id_factory

    def place(self, draft: OrderDraft, *, idempotency_key: str) -> Receipt:
        """Execute one command with safe repeat semantics."""
        if not isinstance(draft, OrderDraft):
            raise TypeError("draft must be an OrderDraft")
        key = _clean_text(idempotency_key, field="idempotency_key")
        existing = self._store.find_by_idempotency_key(key)
        if existing is not None:
            if existing.fingerprint != draft.fingerprint:
                raise ConflictError("idempotency key conflicts with request data")
            return existing

        order_id = _clean_text(self._id_factory(), field="order_id")
        quote = self._pricing.quote(draft)
        if not isinstance(quote, Quote):
            raise TypeError("pricing strategy must return a Quote")
        receipt = Receipt(order_id, key, draft.fingerprint, quote)
        self._store.save(receipt)
        self._events.publish(OrderPlaced(order_id, quote.total_cents, quote.policy_name))
        return receipt


@dataclass(frozen=True, slots=True)
class CreateOrderCommand:
    """Command object carried by a queue, CLI, or HTTP adapter."""

    draft: OrderDraft
    idempotency_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.draft, OrderDraft):
            raise TypeError("draft must be an OrderDraft")
        object.__setattr__(
            self,
            "idempotency_key",
            _clean_text(self.idempotency_key, field="idempotency_key"),
        )


class CreateOrderHandler:
    """Command handler keeps transport parsing outside the service."""

    def __init__(self, service: OrderService) -> None:
        if not isinstance(service, OrderService):
            raise TypeError("service must be an OrderService")
        self._service = service

    def handle(self, command: CreateOrderCommand) -> Receipt:
        if not isinstance(command, CreateOrderCommand):
            raise TypeError("command must be a CreateOrderCommand")
        return self._service.place(
            command.draft,
            idempotency_key=command.idempotency_key,
        )


def run_self_checks() -> None:
    """Check patterns, boundaries, idempotency, and adapter behavior."""
    draft = OrderDraft(
        (
            OrderLine("Python workbook", 1_250, 2),
            OrderLine("Reference card", 500),
        )
    )
    assert draft.subtotal_cents == 3_000
    assert len(draft.fingerprint) == 64

    standard = build_pricing_policy(" STANDARD ")
    sale = build_pricing_policy("sale", percent=10)
    assert standard.quote(draft) == Quote(3_000, 0, 3_000, "standard")
    assert sale.quote(draft) == Quote(3_000, 300, 2_700, "discount-10%")

    store = MemoryOrderStore()
    events = MemoryEventSink()
    ids = iter(("order-001", "should-not-be-used"))
    service = OrderService(
        sale,
        store,
        events,
        id_factory=lambda: next(ids),
    )
    handler = CreateOrderHandler(service)
    command = CreateOrderCommand(draft, "request-001")

    first = handler.handle(command)
    assert first.order_id == "order-001"
    assert first.quote.total_cents == 2_700
    assert handler.handle(command) is first
    assert len(store.receipts) == 1
    assert len(events.events) == 1
    assert next(ids) == "should-not-be-used"

    conflicting = CreateOrderCommand(
        OrderDraft((OrderLine("Different", 3_000),)),
        "request-001",
    )
    try:
        handler.handle(conflicting)
    except ConflictError:
        pass
    else:
        raise AssertionError("conflicting idempotency key must be rejected")

    forwarded: list[OrderPlaced] = []
    CallbackEventSink(forwarded.append).publish(events.events[0])
    assert forwarded == [events.events[0]]
    assert events.events[0].as_json() == (
        '{"event":"order.placed","order_id":"order-001",'
        '"policy":"discount-10%","total_cents":2700}'
    )

    assert isinstance(build_pricing_policy("sale", percent=0), PercentDiscountPolicy)

    invalid_calls = (
        lambda: OrderLine("", 100),
        lambda: OrderLine("Book", True),
        lambda: OrderLine("Book", 100, 0),
        lambda: OrderDraft(()),
        lambda: OrderDraft((object(),)),  # type: ignore[arg-type]
        lambda: Quote(10, 11, 0, "bad"),
        lambda: build_pricing_policy("unknown"),
        lambda: PercentDiscountPolicy(101),
        lambda: MemoryOrderStore().save(object()),  # type: ignore[arg-type]
        lambda: OrderService(object(), store, events, id_factory=lambda: "x"),  # type: ignore[arg-type]
        lambda: handler.handle(object()),  # type: ignore[arg-type]
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a safe architecture demonstration without external services."""
    run_self_checks()

    draft = OrderDraft(
        (
            OrderLine("Architecture workbook", 2_000),
            OrderLine("Testing cards", 500, 2),
        )
    )
    store = MemoryOrderStore()
    events = MemoryEventSink()
    service = OrderService(
        build_pricing_policy("sale", percent=15),
        store,
        events,
        id_factory=lambda: "order-demo-001",
    )
    receipt = CreateOrderHandler(service).handle(
        CreateOrderCommand(draft, "demo-request-001")
    )
    print("Receipt:", receipt)
    print("Events:", [event.as_json() for event in events.events])
    print("Stored receipts:", len(store.receipts))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
