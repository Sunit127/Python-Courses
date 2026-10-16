"""Day 37: Event-driven systems, delivery semantics, and idempotent consumers.

Learning goals
--------------
1. Distinguish commands, events, and event envelopes at service boundaries.
2. Explain at-most-once, at-least-once, and effectively-once processing.
3. Publish committed changes reliably with a transactional outbox.
4. Make consumers idempotent with an inbox and atomic state updates.
5. Classify failures for acknowledgement, retry, or dead-letter handling.

Teaching notes
--------------
- A command asks for work; an event records a fact that already happened. Name
  events in the past tense and keep business meaning separate from broker details.
- An envelope carries stable metadata: identifier, topic, event type, stream key,
  sequence, schema version, and a canonical payload. Its identifier must survive
  retries so consumers can recognize the same logical event.
- At-most-once delivery can lose work but does not redeliver it. At-least-once
  delivery retries unacknowledged work and therefore creates duplicates.
  Effectively-once behavior combines at-least-once delivery with idempotent effects.
- A producer cannot safely commit database state and publish to an independent
  broker as one ordinary transaction. The transactional outbox stores the domain
  change and event together; a relay publishes pending rows afterward.
- A relay may crash after publishing but before marking an outbox row complete.
  Republishing the same event identifier is correct when consumers deduplicate.
- An inbox records each processed event identifier and a content fingerprint.
  Reusing an identifier with different content is a conflict, not a duplicate.
- The inbox record and the consumer's domain update must commit atomically. Recording
  first can lose an effect; recording afterward can repeat an effect after a crash.
- Acknowledge only after the durable effect commits. Retry transient failures with
  bounded attempts. Send permanent poison messages and exhausted retries to a
  dead-letter queue with enough safe context to diagnose and replay them.
- Ordering is normally guaranteed only inside a partition or stream. A per-stream
  sequence detects gaps, but global ordering would reduce availability and scale.
- Keep event payloads small, versioned, non-secret, and backward compatible. Do not
  put credentials or unnecessary personal data in events or dead-letter reasons.
- Production brokers add visibility timeouts, consumer groups, partitions,
  backpressure, retention, and operational monitoring. The in-memory broker below
  models semantics, not broker performance or durability.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, and writes no files.

Practice exercises
------------------
1. Implement EventEnvelope.create() so equivalent payload mappings produce the same
   canonical JSON and content fingerprint.
2. Implement drain_outbox(). Publish before marking a row complete, and leave the row
   pending if publication raises an exception.
3. Implement consume_one(). Acknowledge applied and duplicate events, retry only
   TransientEventError, and dead-letter permanent or exhausted failures.

Solutions appear in the executable code below.

Expert challenge: a reliable order-event pipeline
-------------------------------------------------
Extend the example into a partitioned order pipeline. Persist orders, outbox rows,
consumer inbox rows, inventory reservations, and emitted follow-up events. Add a
relay lease, bounded batches, retry backoff with jitter, dead-letter replay, schema
upcasters, graceful shutdown, and metrics for lag, retries, duplicates, and poison
messages. Prove correctness across crash windows and two competing workers.

Solution guidance
-----------------
1. Put an outbox row in the same database transaction as every order state change.
2. Claim relay batches with a lease or database locking; use a stable event ID on
   every publish attempt and make publishing safe to repeat.
3. Partition by aggregate ID when events for one aggregate require ordering.
4. In the consumer transaction, check the inbox fingerprint, enforce sequence, apply
   the domain update, add follow-up outbox rows, and finally record the inbox entry.
5. Commit before acknowledging. If the process stops before acknowledgement, let the
   broker redeliver and let the inbox turn the repeat into a no-op.
6. Retry timeouts and temporary dependency failures. Dead-letter invalid schemas,
   impossible domain values, and exhausted transient failures.
7. Replay dead letters through the same validation and idempotency path; never edit
   production state manually merely to make a message disappear.
8. Test duplicate publish, crash-after-publish, crash-before-ack, identifier conflict,
   sequence gaps, poison messages, retry exhaustion, and concurrent consumers.

"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from threading import RLock
from typing import Protocol, cast


_JSON_OBJECT = dict[str, object]


class EventSystemError(Exception):
    """Base class for expected event-system failures."""


class ValidationError(EventSystemError, ValueError):
    """Input violates an event or domain boundary."""


class ConflictError(EventSystemError):
    """A stable identifier was reused for different content."""


class TransientEventError(EventSystemError):
    """Processing may succeed when the same event is retried."""


class PermanentEventError(EventSystemError):
    """Retrying unchanged input cannot make processing succeed."""


def _clean_text(value: str, *, field: str, maximum: int = 120) -> str:
    """Return bounded non-empty text for identifiers and routing metadata."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum:
        raise ValidationError(f"{field} must be bounded non-empty text")
    return cleaned


def _positive_int(value: int, *, field: str) -> int:
    """Return a positive integer while rejecting Boolean integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field} must be positive")
    return value


def _non_negative_int(value: int, *, field: str) -> int:
    """Return a non-negative integer while rejecting Boolean integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValidationError(f"{field} must be non-negative")
    return value


def _validate_json(value: object, *, path: str = "$") -> None:
    """Reject non-JSON and non-finite values before crossing a boundary."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{path} must contain finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be text")
            if not key or len(key) > 120:
                raise ValidationError(f"{path} keys must be bounded non-empty text")
            _validate_json(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _canonical_payload(payload: Mapping[str, object]) -> str:
    """Return deterministic JSON for a payload mapping."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    copied = dict(payload)
    _validate_json(copied)
    return json.dumps(
        copied,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Immutable routing metadata plus a canonical JSON object payload."""

    event_id: str
    topic: str
    event_type: str
    stream_id: str
    sequence: int
    schema_version: int
    payload_json: str

    def __post_init__(self) -> None:
        for field in ("event_id", "topic", "event_type", "stream_id"):
            object.__setattr__(
                self,
                field,
                _clean_text(getattr(self, field), field=field),
            )
        _positive_int(self.sequence, field="sequence")
        _positive_int(self.schema_version, field="schema_version")
        if not isinstance(self.payload_json, str):
            raise TypeError("payload_json must be text")
        try:
            decoded = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValidationError("payload_json must contain valid JSON") from error
        if not isinstance(decoded, dict):
            raise ValidationError("payload_json must contain a JSON object")
        canonical = _canonical_payload(cast(_JSON_OBJECT, decoded))
        if canonical != self.payload_json:
            raise ValidationError("payload_json must use canonical encoding")

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        topic: str,
        event_type: str,
        stream_id: str,
        sequence: int,
        schema_version: int,
        payload: Mapping[str, object],
    ) -> EventEnvelope:
        """Solution 1: build an envelope from validated canonical JSON."""
        return cls(
            event_id=event_id,
            topic=topic,
            event_type=event_type,
            stream_id=stream_id,
            sequence=sequence,
            schema_version=schema_version,
            payload_json=_canonical_payload(payload),
        )

    @property
    def payload(self) -> _JSON_OBJECT:
        """Decode a fresh object so callers cannot mutate the envelope."""
        decoded = json.loads(self.payload_json)
        return cast(_JSON_OBJECT, decoded)

    @property
    def fingerprint(self) -> str:
        """Hash every observable envelope field to detect identifier conflicts."""
        content = {
            "event_id": self.event_id,
            "topic": self.topic,
            "event_type": self.event_type,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "schema_version": self.schema_version,
            "payload": self.payload,
        }
        encoded = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Order:
    """The producer's source-of-truth record."""

    order_id: str
    product_id: str
    quantity: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "order_id", _clean_text(self.order_id, field="order_id")
        )
        object.__setattr__(
            self, "product_id", _clean_text(self.product_id, field="product_id")
        )
        _positive_int(self.quantity, field="quantity")


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    """An event stored transactionally with producer state."""

    event: EventEnvelope
    published: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.event, EventEnvelope):
            raise TypeError("event must be EventEnvelope")
        if not isinstance(self.published, bool):
            raise TypeError("published must be Boolean")


class MemoryOrderStore:
    """An atomic in-memory model of orders and their transactional outbox."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._event_by_order: dict[str, str] = {}
        self._outbox: dict[str, OutboxRecord] = {}
        self._lock = RLock()

    def place(self, order: Order) -> EventEnvelope:
        """Commit an order and its event together; make repeats idempotent."""
        if not isinstance(order, Order):
            raise TypeError("order must be Order")
        with self._lock:
            existing = self._orders.get(order.order_id)
            if existing is not None:
                if existing != order:
                    raise ConflictError(
                        f"order_id {order.order_id!r} was reused with new content"
                    )
                event_id = self._event_by_order[order.order_id]
                return self._outbox[event_id].event

            event = EventEnvelope.create(
                event_id=f"event.{order.order_id}.placed.v1",
                topic="orders",
                event_type="order.placed",
                stream_id=order.order_id,
                sequence=1,
                schema_version=1,
                payload={
                    "order_id": order.order_id,
                    "product_id": order.product_id,
                    "quantity": order.quantity,
                },
            )
            self._orders[order.order_id] = order
            self._event_by_order[order.order_id] = event.event_id
            self._outbox[event.event_id] = OutboxRecord(event)
            return event

    def pending(self, *, limit: int = 100) -> tuple[OutboxRecord, ...]:
        """Return a bounded insertion-ordered snapshot of unpublished rows."""
        size = _positive_int(limit, field="limit")
        with self._lock:
            return tuple(
                record
                for record in self._outbox.values()
                if not record.published
            )[:size]

    def mark_published(self, event_id: str) -> None:
        """Mark a known outbox row only after publication succeeds."""
        cleaned = _clean_text(event_id, field="event_id")
        with self._lock:
            record = self._outbox.get(cleaned)
            if record is None:
                raise ValidationError(f"unknown outbox event {cleaned!r}")
            self._outbox[cleaned] = OutboxRecord(record.event, published=True)

    @property
    def order_count(self) -> int:
        with self._lock:
            return len(self._orders)


class Publisher(Protocol):
    """A minimal publication boundary used by the outbox relay."""

    def __call__(self, event: EventEnvelope) -> None:
        ...


def drain_outbox(
    store: MemoryOrderStore,
    publish: Publisher,
    *,
    limit: int = 100,
) -> tuple[str, ...]:
    """Solution 2: publish pending rows, then mark each successful row."""
    if not isinstance(store, MemoryOrderStore):
        raise TypeError("store must be MemoryOrderStore")
    if not callable(publish):
        raise TypeError("publish must be callable")

    published: list[str] = []
    for record in store.pending(limit=limit):
        publish(record.event)
        store.mark_published(record.event.event_id)
        published.append(record.event.event_id)
    return tuple(published)


@dataclass(frozen=True, slots=True)
class Delivery:
    """One broker delivery attempt for an event."""

    event: EventEnvelope
    attempt: int

    def __post_init__(self) -> None:
        if not isinstance(self.event, EventEnvelope):
            raise TypeError("event must be EventEnvelope")
        _positive_int(self.attempt, field="attempt")


@dataclass(frozen=True, slots=True)
class DeadLetter:
    """A delivery removed from normal processing for operator review."""

    event: EventEnvelope
    attempts: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.event, EventEnvelope):
            raise TypeError("event must be EventEnvelope")
        _positive_int(self.attempts, field="attempts")
        object.__setattr__(
            self,
            "reason",
            _clean_text(self.reason, field="reason", maximum=240),
        )


class InMemoryBroker:
    """A deterministic at-least-once broker model with per-topic queues."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[Delivery]] = {}
        self._dead_letters: list[DeadLetter] = []
        self._lock = RLock()

    def publish(self, event: EventEnvelope) -> None:
        """Enqueue every publish call; duplicate delivery is intentional."""
        if not isinstance(event, EventEnvelope):
            raise TypeError("event must be EventEnvelope")
        with self._lock:
            self._queues.setdefault(event.topic, deque()).append(
                Delivery(event, attempt=1)
            )

    def pull(self, topic: str) -> Delivery | None:
        """Reserve the next delivery by removing it from the ready queue."""
        cleaned = _clean_text(topic, field="topic")
        with self._lock:
            queue = self._queues.get(cleaned)
            return None if not queue else queue.popleft()

    def retry(self, delivery: Delivery) -> None:
        """Redeliver later; this small model requeues at the topic tail."""
        if not isinstance(delivery, Delivery):
            raise TypeError("delivery must be Delivery")
        with self._lock:
            self._queues.setdefault(delivery.event.topic, deque()).append(
                Delivery(delivery.event, delivery.attempt + 1)
            )

    def dead_letter(self, delivery: Delivery, reason: str) -> None:
        if not isinstance(delivery, Delivery):
            raise TypeError("delivery must be Delivery")
        letter = DeadLetter(delivery.event, delivery.attempt, reason)
        with self._lock:
            self._dead_letters.append(letter)

    def pending(self, topic: str) -> int:
        cleaned = _clean_text(topic, field="topic")
        with self._lock:
            return len(self._queues.get(cleaned, ()))

    @property
    def dead_letters(self) -> tuple[DeadLetter, ...]:
        with self._lock:
            return tuple(self._dead_letters)


class ApplyResult(Enum):
    """A durable consumer result that is safe to acknowledge."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"


class MemoryInventoryProjection:
    """Inventory plus inbox committed under one lock as one transaction."""

    def __init__(self, available: Mapping[str, int]) -> None:
        if not isinstance(available, Mapping):
            raise TypeError("available must be a mapping")
        self._available: dict[str, int] = {}
        for raw_product_id, raw_quantity in available.items():
            product_id = _clean_text(raw_product_id, field="product_id")
            quantity = _non_negative_int(raw_quantity, field="available quantity")
            if product_id in self._available:
                raise ValidationError("product identifiers must be unique")
            self._available[product_id] = quantity
        self._inbox: dict[str, str] = {}
        self._last_sequence: dict[str, int] = {}
        self._lock = RLock()

    def apply(self, event: EventEnvelope) -> ApplyResult:
        """Apply an order event and inbox marker atomically."""
        if not isinstance(event, EventEnvelope):
            raise TypeError("event must be EventEnvelope")

        with self._lock:
            previous_fingerprint = self._inbox.get(event.event_id)
            if previous_fingerprint is not None:
                if previous_fingerprint != event.fingerprint:
                    raise ConflictError(
                        f"event_id {event.event_id!r} has conflicting content"
                    )
                return ApplyResult.DUPLICATE

            if event.event_type != "order.placed":
                raise PermanentEventError(
                    f"unsupported event type {event.event_type!r}"
                )
            if event.schema_version != 1:
                raise PermanentEventError(
                    f"unsupported schema version {event.schema_version}"
                )

            expected = self._last_sequence.get(event.stream_id, 0) + 1
            if event.sequence > expected:
                raise TransientEventError(
                    f"stream {event.stream_id!r} is missing sequence {expected}"
                )
            if event.sequence < expected:
                raise PermanentEventError(
                    "stale sequence has no matching inbox record"
                )

            payload = event.payload
            required = ("order_id", "product_id", "quantity")
            missing = [name for name in required if name not in payload]
            if missing:
                raise PermanentEventError(
                    f"payload is missing fields: {', '.join(missing)}"
                )
            order_id = payload["order_id"]
            product_id = payload["product_id"]
            quantity = payload["quantity"]
            if not isinstance(order_id, str) or order_id != event.stream_id:
                raise PermanentEventError("order_id must match stream_id")
            if not isinstance(product_id, str):
                raise PermanentEventError("product_id must be text")
            if (
                isinstance(quantity, bool)
                or not isinstance(quantity, int)
                or quantity <= 0
            ):
                raise PermanentEventError("quantity must be a positive integer")

            product_id = _clean_text(product_id, field="product_id")
            if product_id not in self._available:
                raise PermanentEventError(f"unknown product {product_id!r}")
            remaining = self._available[product_id] - quantity
            if remaining < 0:
                raise PermanentEventError(
                    f"insufficient inventory for {product_id!r}"
                )

            # These mutations model one database transaction.
            self._available[product_id] = remaining
            self._last_sequence[event.stream_id] = event.sequence
            self._inbox[event.event_id] = event.fingerprint
            return ApplyResult.APPLIED

    def available(self, product_id: str) -> int:
        cleaned = _clean_text(product_id, field="product_id")
        with self._lock:
            if cleaned not in self._available:
                raise ValidationError(f"unknown product {cleaned!r}")
            return self._available[cleaned]

    @property
    def processed_count(self) -> int:
        with self._lock:
            return len(self._inbox)


class Settlement(Enum):
    """What the worker did with one broker delivery."""

    ACKED = "acked"
    DUPLICATE = "duplicate"
    RETRIED = "retried"
    DEAD_LETTERED = "dead_lettered"
    IDLE = "idle"


@dataclass(frozen=True, slots=True)
class ConsumeReport:
    """A bounded diagnostic result for one worker iteration."""

    settlement: Settlement
    event_id: str | None
    attempt: int
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.settlement, Settlement):
            raise TypeError("settlement must be Settlement")
        if self.event_id is not None:
            object.__setattr__(
                self, "event_id", _clean_text(self.event_id, field="event_id")
            )
        _non_negative_int(self.attempt, field="attempt")
        object.__setattr__(
            self,
            "detail",
            _clean_text(self.detail, field="detail", maximum=240),
        )


class EventHandler(Protocol):
    """A handler whose success represents a durable committed effect."""

    def __call__(self, event: EventEnvelope) -> ApplyResult:
        ...


def _bounded_reason(error: Exception) -> str:
    text = " ".join(str(error).split())
    name = type(error).__name__
    return f"{name}: {text}"[:240] if text else name


def consume_one(
    broker: InMemoryBroker,
    topic: str,
    handler: EventHandler,
    *,
    max_attempts: int = 3,
) -> ConsumeReport:
    """Solution 3: settle one delivery according to classified failure."""
    if not isinstance(broker, InMemoryBroker):
        raise TypeError("broker must be InMemoryBroker")
    if not callable(handler):
        raise TypeError("handler must be callable")
    attempts = _positive_int(max_attempts, field="max_attempts")
    delivery = broker.pull(topic)
    if delivery is None:
        return ConsumeReport(Settlement.IDLE, None, 0, "no delivery available")

    try:
        result = handler(delivery.event)
    except TransientEventError as error:
        reason = _bounded_reason(error)
        if delivery.attempt < attempts:
            broker.retry(delivery)
            return ConsumeReport(
                Settlement.RETRIED,
                delivery.event.event_id,
                delivery.attempt,
                reason,
            )
        broker.dead_letter(delivery, reason)
        return ConsumeReport(
            Settlement.DEAD_LETTERED,
            delivery.event.event_id,
            delivery.attempt,
            reason,
        )
    except (PermanentEventError, ConflictError) as error:
        reason = _bounded_reason(error)
        broker.dead_letter(delivery, reason)
        return ConsumeReport(
            Settlement.DEAD_LETTERED,
            delivery.event.event_id,
            delivery.attempt,
            reason,
        )

    if result is ApplyResult.DUPLICATE:
        return ConsumeReport(
            Settlement.DUPLICATE,
            delivery.event.event_id,
            delivery.attempt,
            "inbox already contains this event",
        )
    if result is not ApplyResult.APPLIED:
        raise TypeError("handler must return ApplyResult")
    return ConsumeReport(
        Settlement.ACKED,
        delivery.event.event_id,
        delivery.attempt,
        "durable effect committed",
    )


class FlakyHandler:
    """A deterministic adapter that fails before delegation a fixed number of times."""

    def __init__(self, delegate: EventHandler, failures: Mapping[str, int]) -> None:
        if not callable(delegate):
            raise TypeError("delegate must be callable")
        if not isinstance(failures, Mapping):
            raise TypeError("failures must be a mapping")
        self._delegate = delegate
        self._remaining: dict[str, int] = {}
        for raw_event_id, raw_count in failures.items():
            event_id = _clean_text(raw_event_id, field="event_id")
            self._remaining[event_id] = _non_negative_int(
                raw_count, field="failure count"
            )

    def __call__(self, event: EventEnvelope) -> ApplyResult:
        remaining = self._remaining.get(event.event_id, 0)
        if remaining:
            self._remaining[event.event_id] = remaining - 1
            raise TransientEventError("inventory database is temporarily busy")
        return self._delegate(event)


def make_order_event(
    order_id: str,
    product_id: str,
    quantity: int,
    *,
    sequence: int = 1,
    schema_version: int = 1,
    event_id: str | None = None,
) -> EventEnvelope:
    """Create a concise event fixture for examples and tests."""
    order = Order(order_id, product_id, quantity)
    return EventEnvelope.create(
        event_id=event_id or f"event.{order.order_id}.{sequence}",
        topic="orders",
        event_type="order.placed",
        stream_id=order.order_id,
        sequence=sequence,
        schema_version=schema_version,
        payload={
            "order_id": order.order_id,
            "product_id": order.product_id,
            "quantity": order.quantity,
        },
    )


def run_self_checks() -> None:
    """Exercise canonical events, crash windows, retries, poison data, and ordering."""
    left = EventEnvelope.create(
        event_id="event.canonical",
        topic="orders",
        event_type="order.placed",
        stream_id="order-canonical",
        sequence=1,
        schema_version=1,
        payload={"quantity": 2, "product_id": "book", "order_id": "order-canonical"},
    )
    right = EventEnvelope.create(
        event_id="event.canonical",
        topic="orders",
        event_type="order.placed",
        stream_id="order-canonical",
        sequence=1,
        schema_version=1,
        payload={"order_id": "order-canonical", "product_id": "book", "quantity": 2},
    )
    assert left.payload_json == right.payload_json
    assert left.fingerprint == right.fingerprint
    detached = left.payload
    detached["quantity"] = 99
    assert left.payload["quantity"] == 2

    store = MemoryOrderStore()
    event = store.place(Order("order-100", "book", 3))
    assert store.place(Order("order-100", "book", 3)) == event
    assert store.order_count == 1
    assert len(store.pending()) == 1
    try:
        store.place(Order("order-100", "book", 4))
    except ConflictError:
        pass
    else:
        raise AssertionError("conflicting order reuse must fail")

    broker = InMemoryBroker()

    # Simulate a relay crash after publish but before mark_published.
    broker.publish(event)
    assert drain_outbox(store, broker.publish) == (event.event_id,)
    assert store.pending() == ()
    assert drain_outbox(store, broker.publish) == ()
    assert broker.pending("orders") == 2

    inventory = MemoryInventoryProjection({"book": 10, "pen": 5})
    first = consume_one(broker, "orders", inventory.apply)
    duplicate = consume_one(broker, "orders", inventory.apply)
    assert first.settlement is Settlement.ACKED
    assert duplicate.settlement is Settlement.DUPLICATE
    assert inventory.available("book") == 7
    assert inventory.processed_count == 1
    assert consume_one(broker, "orders", inventory.apply).settlement is Settlement.IDLE

    retry_event = make_order_event("order-200", "pen", 2)
    broker.publish(retry_event)
    flaky = FlakyHandler(inventory.apply, {retry_event.event_id: 1})
    retried = consume_one(broker, "orders", flaky, max_attempts=2)
    recovered = consume_one(broker, "orders", flaky, max_attempts=2)
    assert retried.settlement is Settlement.RETRIED
    assert recovered.settlement is Settlement.ACKED
    assert recovered.attempt == 2
    assert inventory.available("pen") == 3

    exhausted_event = make_order_event("order-300", "pen", 1)
    broker.publish(exhausted_event)
    always_fails = FlakyHandler(
        inventory.apply, {exhausted_event.event_id: 10}
    )
    assert (
        consume_one(broker, "orders", always_fails, max_attempts=2).settlement
        is Settlement.RETRIED
    )
    exhausted = consume_one(broker, "orders", always_fails, max_attempts=2)
    assert exhausted.settlement is Settlement.DEAD_LETTERED
    assert broker.dead_letters[-1].attempts == 2
    assert inventory.available("pen") == 3

    unsupported = make_order_event(
        "order-400", "book", 1, schema_version=2
    )
    broker.publish(unsupported)
    poison = consume_one(broker, "orders", inventory.apply)
    assert poison.settlement is Settlement.DEAD_LETTERED
    assert "unsupported schema" in broker.dead_letters[-1].reason

    gap = make_order_event("order-500", "book", 1, sequence=2)
    broker.publish(gap)
    gap_first = consume_one(broker, "orders", inventory.apply, max_attempts=2)
    gap_last = consume_one(broker, "orders", inventory.apply, max_attempts=2)
    assert gap_first.settlement is Settlement.RETRIED
    assert gap_last.settlement is Settlement.DEAD_LETTERED

    conflict = EventEnvelope.create(
        event_id=event.event_id,
        topic=event.topic,
        event_type=event.event_type,
        stream_id=event.stream_id,
        sequence=event.sequence,
        schema_version=event.schema_version,
        payload={
            "order_id": "order-100",
            "product_id": "book",
            "quantity": 4,
        },
    )
    broker.publish(conflict)
    conflicted = consume_one(broker, "orders", inventory.apply)
    assert conflicted.settlement is Settlement.DEAD_LETTERED
    assert "ConflictError" in broker.dead_letters[-1].reason
    assert inventory.available("book") == 7

    failed_store = MemoryOrderStore()
    failed_store.place(Order("order-fail", "book", 1))

    def fail_publish(_: EventEnvelope) -> None:
        raise RuntimeError("broker unavailable")

    try:
        drain_outbox(failed_store, fail_publish)
    except RuntimeError:
        pass
    else:
        raise AssertionError("publication failure must propagate")
    assert len(failed_store.pending()) == 1

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: EventEnvelope.create(
            event_id="x",
            topic="orders",
            event_type="order.placed",
            stream_id="order-x",
            sequence=0,
            schema_version=1,
            payload={},
        ),
        lambda: EventEnvelope.create(
            event_id="x",
            topic="orders",
            event_type="order.placed",
            stream_id="order-x",
            sequence=1,
            schema_version=1,
            payload={"bad": float("nan")},
        ),
        lambda: EventEnvelope.create(
            event_id="x",
            topic="orders",
            event_type="order.placed",
            stream_id="order-x",
            sequence=1,
            schema_version=1,
            payload={"bad": {1, 2}},
        ),
        lambda: Order("order-x", "book", True),
        lambda: store.pending(limit=0),
        lambda: broker.publish("not an event"),  # type: ignore[arg-type]
        lambda: inventory.available("missing"),
        lambda: consume_one(broker, "orders", inventory.apply, max_attempts=0),
        lambda: FlakyHandler(inventory.apply, {"event": -1}),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError, EventSystemError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a deterministic outbox-to-idempotent-consumer demonstration."""
    run_self_checks()

    orders = MemoryOrderStore()
    broker = InMemoryBroker()
    inventory = MemoryInventoryProjection({"course": 12})

    event = orders.place(Order("demo-001", "course", 2))
    published = drain_outbox(orders, broker.publish)
    report = consume_one(broker, "orders", inventory.apply)

    print("Published event:", event.event_id)
    print("Outbox batch:", published)
    print("Consumer settlement:", report.settlement.value)
    print("Remaining inventory:", inventory.available("course"))
    print("Dead letters:", len(broker.dead_letters))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
