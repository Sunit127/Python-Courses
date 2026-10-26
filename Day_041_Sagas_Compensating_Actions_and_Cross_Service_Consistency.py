"""Day 41: Sagas, compensating actions, and cross-service consistency.

Learning goals
--------------
1. Explain why one ACID transaction usually cannot span independent services.
2. Model a long-running business operation as durable, idempotent saga steps.
3. Compensate completed steps in reverse order after a later step fails.
4. Resume safely after a crash between a side effect and progress recording.
5. Escalate incomplete compensation for explicit operator intervention.

Teaching notes
--------------
- A saga is a sequence of local transactions. Each service commits its own state;
  the overall workflow reaches consistency through later steps or compensations.
- Compensation is a new business action, not a time machine. A refund does not
  erase a charge from audit history, and releasing stock cannot guarantee that a
  customer never observed the temporary reservation.
- Every forward action and compensation needs a stable idempotency key. The key
  must identify one logical effect and reject reuse for different parameters.
- Record saga intent before the first effect. After each successful action, record
  progress durably. If a process dies in the gap, replay the action with the same
  key and then record its already-applied result.
- Compensate in reverse completion order because later steps may depend on earlier
  ones. Do not compensate a step that never completed.
- Business rejection and infrastructure uncertainty are different. A known
  rejection can start compensation; a timeout may require querying or replaying
  the idempotent operation before deciding what happened.
- Compensation can fail too. Keep retry metadata, continue independent cleanup
  where safe, expose the remaining obligations, and route exhausted work to a
  human-owned queue with a runbook.
- Orchestration centralizes progress and recovery policy. Choreography can reduce
  central coupling but makes global state, ordering, and debugging harder. Choose
  deliberately rather than treating either style as universally better.
- In production, replace this in-memory store with transactional persistence and
  protect concurrent runners with compare-and-set versions, leases, or fencing
  tokens from Day 40. Emit outbox events with state changes when other services
  must observe saga progress.
- Sagas provide eventual consistency, not serializable isolation. Define visible
  intermediate states, invariants, time limits, cancellation rules, and which
  compensations are semantically acceptable with product and operations teams.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, writes no files,
and uses deterministic in-memory services.

Practice exercises
------------------
1. Implement compensation_order(completed, compensated). Validate unique step
   names and return uncompensated completed steps in reverse order.
2. Implement operation_key(saga_id, step_name, phase). Normalize identifiers and
   allow only the phases "forward" and "compensate".
3. Implement requires_manual_review(status, failures). Return True when a saga is
   marked for intervention or has unresolved compensation failures.

Solutions appear in the executable code below.

Expert challenge: production order-fulfillment saga
---------------------------------------------------
Build a persisted saga coordinator using PostgreSQL, an outbox, and workers that
claim versioned records with leases and fencing tokens. Integrate inventory,
payment, shipping, and notification adapters through explicit contracts. Add
timeouts, retry budgets with jitter, cancellation, poison-message quarantine,
operator retry/override commands, immutable audit history, tracing, dashboards,
and alerts for stuck sagas. Prove idempotency under lost responses and duplicate
delivery. Test crashes at every state transition, two coordinators racing,
dependency outages, compensation rejection, schema upgrades during active sagas,
and disaster recovery from backups. Document consistency guarantees, privacy and
security boundaries, service-level objectives, and the operator runbook.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from threading import RLock


class SagaError(RuntimeError):
    """Base class for saga coordination failures."""


class ValidationError(SagaError, ValueError):
    """Raised when public data violates a workflow invariant."""


class OperationConflict(SagaError):
    """Raised when an idempotency key is reused for different work."""


class SagaConflict(SagaError):
    """Raised when a saga id is reused with different input."""


class ServiceRejected(SagaError):
    """Raised when a participant rejects a known business operation."""


class SimulatedCrash(SagaError):
    """Raised at a deliberate crash window in the demonstration."""


class SagaStatus(Enum):
    """Durable high-level state of an orchestrated saga."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    MANUAL_INTERVENTION = "manual_intervention"


def _clean_text(value: object, *, field: str, limit: int = 120) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValidationError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise ValidationError(f"{field} must be at most {limit} characters")
    return cleaned


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field} must be positive")
    return value


def _unique_names(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    names = tuple(_clean_text(value, field=field) for value in values)
    if len(names) != len(set(names)):
        raise ValidationError(f"{field} must not contain duplicates")
    return names


@dataclass(frozen=True, slots=True)
class OrderContext:
    """Validated immutable input shared by every order-saga step."""

    order_id: str
    sku: str
    quantity: int
    amount_cents: int
    address: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "order_id",
            _clean_text(self.order_id, field="order_id"),
        )
        object.__setattr__(self, "sku", _clean_text(self.sku, field="sku"))
        object.__setattr__(
            self,
            "quantity",
            _positive_int(self.quantity, field="quantity"),
        )
        object.__setattr__(
            self,
            "amount_cents",
            _positive_int(self.amount_cents, field="amount_cents"),
        )
        object.__setattr__(
            self,
            "address",
            _clean_text(self.address, field="address", limit=240),
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


Action = Callable[[OrderContext, str], None]


@dataclass(frozen=True, slots=True)
class SagaStep:
    """One forward action and the business action that compensates it."""

    name: str
    action: Action
    compensate: Action

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _clean_text(self.name, field="step.name"),
        )
        if not callable(self.action):
            raise ValidationError("step.action must be callable")
        if not callable(self.compensate):
            raise ValidationError("step.compensate must be callable")


@dataclass(slots=True)
class _SagaRecord:
    saga_id: str
    context_fingerprint: str
    status: SagaStatus = SagaStatus.RUNNING
    completed_steps: list[str] = field(default_factory=list)
    compensated_steps: list[str] = field(default_factory=list)
    failure: str | None = None
    compensation_failures: dict[str, str] = field(default_factory=dict)
    revision: int = 0


@dataclass(frozen=True, slots=True)
class SagaSnapshot:
    """Immutable diagnostic view returned to callers and operators."""

    saga_id: str
    status: SagaStatus
    completed_steps: tuple[str, ...]
    compensated_steps: tuple[str, ...]
    failure: str | None
    compensation_failures: tuple[tuple[str, str], ...]
    revision: int


class InMemorySagaStore:
    """Thread-safe teaching store; production storage must be durable."""

    def __init__(self) -> None:
        self._records: dict[str, _SagaRecord] = {}
        self._lock = RLock()

    def get_or_create(
        self,
        saga_id: str,
        context_fingerprint: str,
    ) -> _SagaRecord:
        identifier = _clean_text(saga_id, field="saga_id")
        fingerprint = _clean_text(
            context_fingerprint,
            field="context_fingerprint",
        )
        with self._lock:
            existing = self._records.get(identifier)
            if existing is not None:
                if existing.context_fingerprint != fingerprint:
                    raise SagaConflict(
                        "saga_id was already used with different input"
                    )
                return existing
            record = _SagaRecord(identifier, fingerprint)
            self._records[identifier] = record
            return record

    def mutate(self, record: _SagaRecord, change: Callable[[], None]) -> None:
        """Apply one state change and advance its observable revision."""
        if not callable(change):
            raise ValidationError("change must be callable")
        with self._lock:
            current = self._records.get(record.saga_id)
            if current is not record:
                raise SagaConflict("record is not current")
            change()
            record.revision += 1

    def snapshot(self, saga_id: str) -> SagaSnapshot:
        identifier = _clean_text(saga_id, field="saga_id")
        with self._lock:
            try:
                record = self._records[identifier]
            except KeyError as error:
                raise KeyError(identifier) from error
            return SagaSnapshot(
                saga_id=record.saga_id,
                status=record.status,
                completed_steps=tuple(record.completed_steps),
                compensated_steps=tuple(record.compensated_steps),
                failure=record.failure,
                compensation_failures=tuple(
                    sorted(record.compensation_failures.items())
                ),
                revision=record.revision,
            )


class IdempotentParticipant:
    """Shared helper that applies a stable operation key at most once."""

    def __init__(self, events: list[str]) -> None:
        self._operations: dict[str, tuple[object, ...]] = {}
        self._events = events
        self._lock = RLock()

    def _apply_once(
        self,
        operation_id: str,
        signature: tuple[object, ...],
        effect: Callable[[], None],
        event: str,
    ) -> bool:
        identifier = _clean_text(operation_id, field="operation_id")
        with self._lock:
            previous = self._operations.get(identifier)
            if previous is not None:
                if previous != signature:
                    raise OperationConflict(
                        "operation_id was already used for different work"
                    )
                return False
            effect()
            self._operations[identifier] = signature
            self._events.append(event)
            return True

    def applied_count(self) -> int:
        with self._lock:
            return len(self._operations)


class InventoryService(IdempotentParticipant):
    """Local inventory transaction with an idempotent release."""

    def __init__(self, stock: dict[str, int], events: list[str]) -> None:
        super().__init__(events)
        self._stock = dict(stock)
        self._reservations: dict[str, tuple[str, int]] = {}

    def reserve(
        self,
        operation_id: str,
        order_id: str,
        sku: str,
        quantity: int,
    ) -> None:
        order = _clean_text(order_id, field="order_id")
        item = _clean_text(sku, field="sku")
        amount = _positive_int(quantity, field="quantity")

        def effect() -> None:
            available = self._stock.get(item, 0)
            if available < amount:
                raise ServiceRejected(f"insufficient stock for {item}")
            if order in self._reservations:
                raise ServiceRejected("order already has a reservation")
            self._stock[item] = available - amount
            self._reservations[order] = (item, amount)

        self._apply_once(
            operation_id,
            ("reserve", order, item, amount),
            effect,
            f"inventory.reserve:{order}",
        )

    def release(
        self,
        operation_id: str,
        order_id: str,
        sku: str,
        quantity: int,
    ) -> None:
        order = _clean_text(order_id, field="order_id")
        item = _clean_text(sku, field="sku")
        amount = _positive_int(quantity, field="quantity")

        def effect() -> None:
            if self._reservations.get(order) != (item, amount):
                raise ServiceRejected("matching reservation does not exist")
            del self._reservations[order]
            self._stock[item] = self._stock.get(item, 0) + amount

        self._apply_once(
            operation_id,
            ("release", order, item, amount),
            effect,
            f"inventory.release:{order}",
        )

    def available(self, sku: str) -> int:
        item = _clean_text(sku, field="sku")
        return self._stock.get(item, 0)

    def is_reserved(self, order_id: str) -> bool:
        order = _clean_text(order_id, field="order_id")
        return order in self._reservations


class PaymentService(IdempotentParticipant):
    """Local payment ledger that preserves charge and refund history."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.reject_charges = False
        self.reject_refunds = False
        self._charges: dict[str, int] = {}
        self._refunded: set[str] = set()

    def charge(
        self,
        operation_id: str,
        order_id: str,
        amount_cents: int,
    ) -> None:
        order = _clean_text(order_id, field="order_id")
        amount = _positive_int(amount_cents, field="amount_cents")

        def effect() -> None:
            if self.reject_charges:
                raise ServiceRejected("payment was declined")
            previous = self._charges.get(order)
            if previous is not None and previous != amount:
                raise ServiceRejected("order has a different charge")
            self._charges[order] = amount

        self._apply_once(
            operation_id,
            ("charge", order, amount),
            effect,
            f"payment.charge:{order}",
        )

    def refund(
        self,
        operation_id: str,
        order_id: str,
        amount_cents: int,
    ) -> None:
        order = _clean_text(order_id, field="order_id")
        amount = _positive_int(amount_cents, field="amount_cents")

        def effect() -> None:
            if self.reject_refunds:
                raise ServiceRejected("refund provider unavailable")
            if self._charges.get(order) != amount:
                raise ServiceRejected("matching charge does not exist")
            self._refunded.add(order)

        self._apply_once(
            operation_id,
            ("refund", order, amount),
            effect,
            f"payment.refund:{order}",
        )

    def is_charged(self, order_id: str) -> bool:
        order = _clean_text(order_id, field="order_id")
        return order in self._charges

    def is_refunded(self, order_id: str) -> bool:
        order = _clean_text(order_id, field="order_id")
        return order in self._refunded


class ShippingService(IdempotentParticipant):
    """Local shipment transaction with cancellation compensation."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.reject_schedules = False
        self._shipments: dict[str, str] = {}
        self._cancelled: set[str] = set()

    def schedule(
        self,
        operation_id: str,
        order_id: str,
        address: str,
    ) -> None:
        order = _clean_text(order_id, field="order_id")
        destination = _clean_text(address, field="address", limit=240)

        def effect() -> None:
            if self.reject_schedules:
                raise ServiceRejected("carrier rejected the destination")
            previous = self._shipments.get(order)
            if previous is not None and previous != destination:
                raise ServiceRejected("order has another destination")
            self._shipments[order] = destination

        self._apply_once(
            operation_id,
            ("schedule", order, destination),
            effect,
            f"shipping.schedule:{order}",
        )

    def cancel(
        self,
        operation_id: str,
        order_id: str,
        address: str,
    ) -> None:
        order = _clean_text(order_id, field="order_id")
        destination = _clean_text(address, field="address", limit=240)

        def effect() -> None:
            if self._shipments.get(order) != destination:
                raise ServiceRejected("matching shipment does not exist")
            self._cancelled.add(order)

        self._apply_once(
            operation_id,
            ("cancel", order, destination),
            effect,
            f"shipping.cancel:{order}",
        )

    def is_scheduled(self, order_id: str) -> bool:
        order = _clean_text(order_id, field="order_id")
        return order in self._shipments

    def is_cancelled(self, order_id: str) -> bool:
        order = _clean_text(order_id, field="order_id")
        return order in self._cancelled


def compensation_order(
    completed: Iterable[str],
    compensated: Iterable[str],
) -> tuple[str, ...]:
    """Solution 1: list remaining compensations in reverse completion order."""
    completed_names = _unique_names(completed, field="completed")
    compensated_names = _unique_names(compensated, field="compensated")
    unknown = set(compensated_names) - set(completed_names)
    if unknown:
        raise ValidationError("compensated steps must also be completed")
    already_done = set(compensated_names)
    return tuple(
        name for name in reversed(completed_names) if name not in already_done
    )


def operation_key(saga_id: str, step_name: str, phase: str) -> str:
    """Solution 2: build a stable key for one logical participant effect."""
    saga = _clean_text(saga_id, field="saga_id")
    step = _clean_text(step_name, field="step_name")
    normalized_phase = _clean_text(phase, field="phase").lower()
    if normalized_phase not in {"forward", "compensate"}:
        raise ValidationError("phase must be 'forward' or 'compensate'")
    return f"{saga}:{step}:{normalized_phase}"


def requires_manual_review(
    status: SagaStatus,
    failures: Iterable[tuple[str, str]],
) -> bool:
    """Solution 3: identify a saga with unresolved cleanup obligations."""
    if not isinstance(status, SagaStatus):
        raise ValidationError("status must be SagaStatus")
    failure_items = tuple(failures)
    for item in failure_items:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
        ):
            raise ValidationError("failures must contain (step, reason) pairs")
    return status is SagaStatus.MANUAL_INTERVENTION or bool(failure_items)


def _bounded_error(error: Exception) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    return f"{type(error).__name__}: {message}"[:300]


class SagaOrchestrator:
    """Execute, resume, compensate, and inspect ordered saga steps."""

    def __init__(self, store: InMemorySagaStore) -> None:
        if not isinstance(store, InMemorySagaStore):
            raise ValidationError("store must be InMemorySagaStore")
        self._store = store

    def execute(
        self,
        saga_id: str,
        context: OrderContext,
        steps: Iterable[SagaStep],
        *,
        crash_after_action: str | None = None,
    ) -> SagaSnapshot:
        """Run forward from durable progress, or continue compensation."""
        if not isinstance(context, OrderContext):
            raise ValidationError("context must be OrderContext")
        ordered_steps = self._validated_steps(steps)
        step_by_name = {step.name: step for step in ordered_steps}
        if crash_after_action is not None:
            crash_name = _clean_text(
                crash_after_action,
                field="crash_after_action",
            )
            if crash_name not in step_by_name:
                raise ValidationError("crash_after_action must name a step")
        else:
            crash_name = None

        record = self._store.get_or_create(saga_id, context.fingerprint())
        if record.status in {
            SagaStatus.COMPENSATING,
            SagaStatus.MANUAL_INTERVENTION,
        }:
            return self._compensate(record, context, step_by_name)
        if record.status in {
            SagaStatus.COMPLETED,
            SagaStatus.COMPENSATED,
        }:
            return self._store.snapshot(record.saga_id)

        for step in ordered_steps:
            if step.name in record.completed_steps:
                continue
            key = operation_key(record.saga_id, step.name, "forward")
            try:
                step.action(context, key)
            except Exception as error:
                self._store.mutate(
                    record,
                    lambda error=error: self._start_compensation(
                        record,
                        error,
                    ),
                )
                return self._compensate(record, context, step_by_name)

            if step.name == crash_name:
                raise SimulatedCrash(
                    f"crashed after {step.name!r} action before progress record"
                )

            self._store.mutate(
                record,
                lambda name=step.name: record.completed_steps.append(name),
            )

        self._store.mutate(
            record,
            lambda: setattr(record, "status", SagaStatus.COMPLETED),
        )
        return self._store.snapshot(record.saga_id)

    def retry_compensation(
        self,
        saga_id: str,
        context: OrderContext,
        steps: Iterable[SagaStep],
    ) -> SagaSnapshot:
        """Retry only compensation obligations that remain unresolved."""
        if not isinstance(context, OrderContext):
            raise ValidationError("context must be OrderContext")
        ordered_steps = self._validated_steps(steps)
        step_by_name = {step.name: step for step in ordered_steps}
        record = self._store.get_or_create(saga_id, context.fingerprint())
        if record.status not in {
            SagaStatus.COMPENSATING,
            SagaStatus.MANUAL_INTERVENTION,
        }:
            return self._store.snapshot(record.saga_id)
        return self._compensate(record, context, step_by_name)

    @staticmethod
    def _validated_steps(steps: Iterable[SagaStep]) -> tuple[SagaStep, ...]:
        try:
            result = tuple(steps)
        except TypeError as error:
            raise ValidationError("steps must be iterable") from error
        if not result:
            raise ValidationError("steps must not be empty")
        if not all(isinstance(step, SagaStep) for step in result):
            raise ValidationError("steps must contain SagaStep values")
        _unique_names((step.name for step in result), field="step.name")
        return result

    @staticmethod
    def _start_compensation(
        record: _SagaRecord,
        error: Exception,
    ) -> None:
        record.failure = _bounded_error(error)
        record.status = SagaStatus.COMPENSATING
        record.compensation_failures.clear()

    def _compensate(
        self,
        record: _SagaRecord,
        context: OrderContext,
        step_by_name: dict[str, SagaStep],
    ) -> SagaSnapshot:
        self._store.mutate(
            record,
            lambda: (
                setattr(record, "status", SagaStatus.COMPENSATING),
                record.compensation_failures.clear(),
            ),
        )
        failures: dict[str, str] = {}

        for name in compensation_order(
            record.completed_steps,
            record.compensated_steps,
        ):
            step = step_by_name.get(name)
            if step is None:
                failures[name] = "ValidationError: completed step is undefined"
                continue
            key = operation_key(record.saga_id, name, "compensate")
            try:
                step.compensate(context, key)
            except Exception as error:
                failures[name] = _bounded_error(error)
                continue
            self._store.mutate(
                record,
                lambda name=name: record.compensated_steps.append(name),
            )

        def finish() -> None:
            record.compensation_failures.update(failures)
            if compensation_order(
                record.completed_steps,
                record.compensated_steps,
            ):
                record.status = SagaStatus.MANUAL_INTERVENTION
            else:
                record.status = SagaStatus.COMPENSATED

        self._store.mutate(record, finish)
        return self._store.snapshot(record.saga_id)


def order_steps(
    inventory: InventoryService,
    payments: PaymentService,
    shipping: ShippingService,
) -> tuple[SagaStep, ...]:
    """Build explicit participant adapters for the order saga."""
    if not isinstance(inventory, InventoryService):
        raise ValidationError("inventory must be InventoryService")
    if not isinstance(payments, PaymentService):
        raise ValidationError("payments must be PaymentService")
    if not isinstance(shipping, ShippingService):
        raise ValidationError("shipping must be ShippingService")

    return (
        SagaStep(
            "reserve_inventory",
            lambda context, key: inventory.reserve(
                key,
                context.order_id,
                context.sku,
                context.quantity,
            ),
            lambda context, key: inventory.release(
                key,
                context.order_id,
                context.sku,
                context.quantity,
            ),
        ),
        SagaStep(
            "charge_payment",
            lambda context, key: payments.charge(
                key,
                context.order_id,
                context.amount_cents,
            ),
            lambda context, key: payments.refund(
                key,
                context.order_id,
                context.amount_cents,
            ),
        ),
        SagaStep(
            "schedule_shipping",
            lambda context, key: shipping.schedule(
                key,
                context.order_id,
                context.address,
            ),
            lambda context, key: shipping.cancel(
                key,
                context.order_id,
                context.address,
            ),
        ),
    )


def _expect_error(
    callback: Callable[[], object],
    expected: type[BaseException],
) -> None:
    try:
        callback()
    except expected:
        pass
    else:
        raise AssertionError(f"expected {expected.__name__}")


def run_self_checks() -> None:
    """Verify success, replay, crash recovery, compensation, and escalation."""
    events: list[str] = []
    inventory = InventoryService({"COURSE": 10}, events)
    payments = PaymentService(events)
    shipping = ShippingService(events)
    steps = order_steps(inventory, payments, shipping)
    store = InMemorySagaStore()
    orchestrator = SagaOrchestrator(store)
    context = OrderContext(
        "order-41-a",
        "COURSE",
        2,
        4_100,
        "Kathmandu",
    )

    completed = orchestrator.execute("saga-41-a", context, steps)
    assert completed.status is SagaStatus.COMPLETED
    assert completed.completed_steps == (
        "reserve_inventory",
        "charge_payment",
        "schedule_shipping",
    )
    assert inventory.available("COURSE") == 8
    assert payments.is_charged("order-41-a")
    assert shipping.is_scheduled("order-41-a")

    event_count = len(events)
    replay = orchestrator.execute("saga-41-a", context, steps)
    assert replay == completed
    assert len(events) == event_count

    crash_events: list[str] = []
    crash_inventory = InventoryService({"COURSE": 5}, crash_events)
    crash_payments = PaymentService(crash_events)
    crash_shipping = ShippingService(crash_events)
    crash_steps = order_steps(
        crash_inventory,
        crash_payments,
        crash_shipping,
    )
    crash_store = InMemorySagaStore()
    crash_orchestrator = SagaOrchestrator(crash_store)
    crash_context = OrderContext(
        "order-41-crash",
        "COURSE",
        1,
        2_000,
        "Pokhara",
    )

    _expect_error(
        lambda: crash_orchestrator.execute(
            "saga-41-crash",
            crash_context,
            crash_steps,
            crash_after_action="charge_payment",
        ),
        SimulatedCrash,
    )
    interrupted = crash_store.snapshot("saga-41-crash")
    assert interrupted.status is SagaStatus.RUNNING
    assert interrupted.completed_steps == ("reserve_inventory",)
    assert crash_payments.applied_count() == 1

    resumed = crash_orchestrator.execute(
        "saga-41-crash",
        crash_context,
        crash_steps,
    )
    assert resumed.status is SagaStatus.COMPLETED
    assert crash_payments.applied_count() == 1
    assert crash_shipping.is_scheduled("order-41-crash")

    failed_events: list[str] = []
    failed_inventory = InventoryService({"COURSE": 4}, failed_events)
    failed_payments = PaymentService(failed_events)
    failed_shipping = ShippingService(failed_events)
    failed_shipping.reject_schedules = True
    failed_steps = order_steps(
        failed_inventory,
        failed_payments,
        failed_shipping,
    )
    failed_orchestrator = SagaOrchestrator(InMemorySagaStore())
    failed_context = OrderContext(
        "order-41-fail",
        "COURSE",
        2,
        3_000,
        "Invalid Zone",
    )

    compensated = failed_orchestrator.execute(
        "saga-41-fail",
        failed_context,
        failed_steps,
    )
    assert compensated.status is SagaStatus.COMPENSATED
    assert compensated.compensated_steps == (
        "charge_payment",
        "reserve_inventory",
    )
    assert failed_events[-2:] == [
        "payment.refund:order-41-fail",
        "inventory.release:order-41-fail",
    ]
    assert failed_inventory.available("COURSE") == 4
    assert failed_payments.is_refunded("order-41-fail")

    manual_events: list[str] = []
    manual_inventory = InventoryService({"COURSE": 3}, manual_events)
    manual_payments = PaymentService(manual_events)
    manual_shipping = ShippingService(manual_events)
    manual_payments.reject_refunds = True
    manual_shipping.reject_schedules = True
    manual_steps = order_steps(
        manual_inventory,
        manual_payments,
        manual_shipping,
    )
    manual_store = InMemorySagaStore()
    manual_orchestrator = SagaOrchestrator(manual_store)
    manual_context = OrderContext(
        "order-41-manual",
        "COURSE",
        1,
        1_500,
        "Unavailable Route",
    )

    manual = manual_orchestrator.execute(
        "saga-41-manual",
        manual_context,
        manual_steps,
    )
    assert manual.status is SagaStatus.MANUAL_INTERVENTION
    assert manual.compensated_steps == ("reserve_inventory",)
    assert tuple(name for name, _ in manual.compensation_failures) == (
        "charge_payment",
    )
    assert requires_manual_review(
        manual.status,
        manual.compensation_failures,
    )
    assert manual_inventory.available("COURSE") == 3

    manual_payments.reject_refunds = False
    recovered = manual_orchestrator.retry_compensation(
        "saga-41-manual",
        manual_context,
        manual_steps,
    )
    assert recovered.status is SagaStatus.COMPENSATED
    assert set(recovered.compensated_steps) == {
        "reserve_inventory",
        "charge_payment",
    }
    assert recovered.compensation_failures == ()
    assert manual_payments.is_refunded("order-41-manual")

    assert compensation_order(("a", "b", "c"), ("c",)) == ("b", "a")
    assert operation_key(" saga-1 ", " charge ", "FORWARD") == (
        "saga-1:charge:forward"
    )
    assert not requires_manual_review(SagaStatus.COMPLETED, ())

    _expect_error(
        lambda: compensation_order(("a", "a"), ()),
        ValidationError,
    )
    _expect_error(
        lambda: compensation_order(("a",), ("b",)),
        ValidationError,
    )
    _expect_error(
        lambda: operation_key("saga", "step", "undo"),
        ValidationError,
    )
    _expect_error(
        lambda: OrderContext("order", "sku", True, 100, "address"),
        ValidationError,
    )
    _expect_error(
        lambda: orchestrator.execute(
            "saga-41-a",
            OrderContext("different", "COURSE", 2, 4_100, "Kathmandu"),
            steps,
        ),
        SagaConflict,
    )
    _expect_error(
        lambda: SagaOrchestrator(InMemorySagaStore()).execute(
            "empty-saga",
            context,
            (),
        ),
        ValidationError,
    )


def main() -> None:
    """Run a failed order saga and display its compensating actions."""
    run_self_checks()

    events: list[str] = []
    inventory = InventoryService({"PYTHON-BOOK": 5}, events)
    payments = PaymentService(events)
    shipping = ShippingService(events)
    shipping.reject_schedules = True

    context = OrderContext(
        order_id="demo-order-41",
        sku="PYTHON-BOOK",
        quantity=1,
        amount_cents=3_500,
        address="Unsupported Delivery Zone",
    )
    orchestrator = SagaOrchestrator(InMemorySagaStore())
    result = orchestrator.execute(
        "demo-saga-41",
        context,
        order_steps(inventory, payments, shipping),
    )

    print("Saga status:", result.status.value)
    print("Completed:", result.completed_steps)
    print("Compensated:", result.compensated_steps)
    print("Participant events:", events)
    print("Stock restored:", inventory.available("PYTHON-BOOK"))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
