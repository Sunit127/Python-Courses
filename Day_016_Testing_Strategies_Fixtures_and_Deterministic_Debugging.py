"""Day 16: Testing strategies, fixtures, and deterministic debugging.

Learning goals
--------------
1. Separate fast unit tests from focused integration tests.
2. Arrange test data with reusable fixtures and clean up every resource.
3. Test normal behavior, boundaries, invalid input, and expected failures.
4. Replace clocks, ID generators, and external services with deterministic fakes.
5. Use descriptive assertions and small reproduction cases to debug failures.

Teaching notes
--------------
- A test is executable evidence about one behavior. Give it a name that states
  the situation and expected result.
- Follow arrange, act, assert: create inputs, perform one behavior, then check
  the observable result. Several assertions are fine when they describe one
  outcome.
- Unit tests isolate a small calculation or object. Integration tests verify
  that collaborating parts, such as code and a temporary file, work together.
- A fixture is repeatable setup shared by tests. unittest.TestCase.setUp runs
  before every test, so mutable state does not leak between tests.
- Test doubles make boundaries controllable. A stub returns prepared data, a
  spy records calls, and a fake provides a small working implementation.
- Inject time, randomness, identifiers, and network-facing collaborators.
  Tests should not depend on today's date, a real payment API, or call order
  from another test.
- Test behavior rather than implementation details. Refactoring should not
  break a test unless the public contract changes.
- When a test fails, reduce it to the smallest reproducible input, read the
  traceback from the first relevant frame, and compare expected with actual.
  Add context with subTest or a precise assertion message.
- Keep production validation in production code. A test suite should prove the
  rules; it should not be the only place where those rules exist.

Run this file with Python 3.10+ to execute the examples and 15 self-tests.
No third-party package, network connection, or persistent file is required.

Practice exercises
------------------
1. Implement shipping_fee_cents: shipping is free at or above a threshold and
   otherwise costs a configured fee.
2. Implement SequentialId, a deterministic callable that returns IDs in order
   and raises when its prepared IDs are exhausted.
3. Implement successful_receipts to select only receipts with a non-empty
   payment reference.

Solutions and tests appear below the main example.

Expert challenge: idempotent checkout
-------------------------------------
Extend CheckoutService with an idempotency key so a retried request cannot
charge a customer twice. The same key and same cart should return the original
receipt; reusing a key for different input should raise a domain error.

Solution guidance
-----------------
1. Validate the key at the application boundary.
2. Store receipts by key behind a narrow repository protocol.
3. Look up the key before calling the payment gateway.
4. Save a cart fingerprint with the receipt and compare it on repeated calls.
5. Test the first request, identical retry, conflicting retry, gateway failure,
   and repository failure. Assert the exact number of gateway calls.
6. In production, enforce uniqueness in durable storage and make the save and
   charge workflow resilient to crashes; an in-memory check alone is not enough.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Protocol, TextIO
import sys
import unittest


def _clean_text(value: str, field_name: str) -> str:
    """Return stripped, non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _non_negative_int(value: int, field_name: str) -> int:
    """Accept an integer of zero or more, but reject Booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


@dataclass(frozen=True, slots=True)
class LineItem:
    """A validated cart line represented in integer cents."""

    name: str
    unit_price_cents: int
    quantity: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, "item name"))
        _non_negative_int(self.unit_price_cents, "unit_price_cents")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

    @property
    def subtotal_cents(self) -> int:
        """Return this line's extended price."""
        return self.unit_price_cents * self.quantity


@dataclass(frozen=True, slots=True)
class TotalBreakdown:
    """Observable calculation steps that make failures easier to diagnose."""

    subtotal_cents: int
    discount_cents: int
    total_cents: int


def calculate_total(
    items: Iterable[LineItem],
    discount_percent: int = 0,
) -> TotalBreakdown:
    """Calculate a cart total, rounding percentage discounts half-up."""
    percent = _non_negative_int(discount_percent, "discount_percent")
    if percent > 100:
        raise ValueError("discount_percent cannot exceed 100")

    cart = tuple(items)
    if not cart:
        raise ValueError("cart cannot be empty")
    if not all(isinstance(item, LineItem) for item in cart):
        raise TypeError("items must contain only LineItem values")

    subtotal = sum(item.subtotal_cents for item in cart)
    discount = (subtotal * percent + 50) // 100
    return TotalBreakdown(subtotal, discount, subtotal - discount)


class PaymentGateway(Protocol):
    """Boundary implemented by a real adapter or a deterministic fake."""

    def charge(self, order_id: str, amount_cents: int) -> str:
        """Charge once and return a public payment reference."""


class ReceiptStore(Protocol):
    """Minimal persistence boundary required by CheckoutService."""

    def save(self, receipt: Receipt) -> None:
        """Persist a completed receipt."""


@dataclass(frozen=True, slots=True)
class Receipt:
    """Result returned after a successful charge."""

    order_id: str
    total_cents: int
    payment_reference: str
    created_at: datetime


class CheckoutError(Exception):
    """Expected failure at the checkout boundary."""


class CheckoutService:
    """Coordinate pure pricing with injected side-effect boundaries."""

    def __init__(
        self,
        gateway: PaymentGateway,
        store: ReceiptStore,
        *,
        id_factory: Callable[[], str],
        clock: Callable[[], datetime],
    ) -> None:
        if not callable(getattr(gateway, "charge", None)):
            raise TypeError("gateway must provide charge()")
        if not callable(getattr(store, "save", None)):
            raise TypeError("store must provide save()")
        if not callable(id_factory) or not callable(clock):
            raise TypeError("id_factory and clock must be callable")
        self._gateway = gateway
        self._store = store
        self._id_factory = id_factory
        self._clock = clock

    def checkout(
        self,
        items: Iterable[LineItem],
        *,
        discount_percent: int = 0,
    ) -> Receipt:
        """Charge and save one validated cart."""
        total = calculate_total(items, discount_percent)
        order_id = _clean_text(self._id_factory(), "generated order ID")
        created_at = self._clock()
        if not isinstance(created_at, datetime):
            raise TypeError("clock must return datetime")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")

        try:
            payment_reference = _clean_text(
                self._gateway.charge(order_id, total.total_cents),
                "payment reference",
            )
        except CheckoutError:
            raise
        except Exception as error:
            raise CheckoutError("payment could not be completed") from error

        receipt = Receipt(
            order_id,
            total.total_cents,
            payment_reference,
            created_at,
        )
        self._store.save(receipt)
        return receipt


class RecordingGateway:
    """Fake gateway that records calls and can return or raise predictably."""

    def __init__(
        self,
        reference: str = "pay-test-001",
        error: Exception | None = None,
    ) -> None:
        self.reference = reference
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def charge(self, order_id: str, amount_cents: int) -> str:
        self.calls.append((order_id, amount_cents))
        if self.error is not None:
            raise self.error
        return self.reference


class MemoryReceiptStore:
    """Fake store with observable in-memory state."""

    def __init__(self) -> None:
        self.receipts: list[Receipt] = []

    def save(self, receipt: Receipt) -> None:
        self.receipts.append(receipt)


def write_receipt_report(
    destination: TextIO,
    receipts: Iterable[Receipt],
) -> int:
    """Write a deterministic CSV-like report and return characters written."""
    rows = ["order_id,total_cents,payment_reference,created_at"]
    for receipt in receipts:
        rows.append(
            ",".join(
                (
                    receipt.order_id,
                    str(receipt.total_cents),
                    receipt.payment_reference,
                    receipt.created_at.isoformat(),
                )
            )
        )
    return destination.write("\n".join(rows) + "\n")


# Practice exercise solutions


def shipping_fee_cents(
    subtotal_cents: int,
    *,
    free_shipping_at_cents: int,
    fee_cents: int,
) -> int:
    """Solution 1: return zero at the threshold, otherwise the fee."""
    subtotal = _non_negative_int(subtotal_cents, "subtotal_cents")
    threshold = _non_negative_int(
        free_shipping_at_cents,
        "free_shipping_at_cents",
    )
    fee = _non_negative_int(fee_cents, "fee_cents")
    return 0 if subtotal >= threshold else fee


class SequentialId:
    """Solution 2: deterministic ID source for tests and examples."""

    def __init__(self, values: Iterable[str]) -> None:
        self._values = iter(values)

    def __call__(self) -> str:
        try:
            value = next(self._values)
        except StopIteration as error:
            raise RuntimeError("no prepared IDs remain") from error
        return _clean_text(value, "prepared ID")


def successful_receipts(receipts: Iterable[Receipt]) -> tuple[Receipt, ...]:
    """Solution 3: retain receipts with a non-empty payment reference."""
    result: list[Receipt] = []
    for receipt in receipts:
        if not isinstance(receipt, Receipt):
            raise TypeError("receipts must contain only Receipt values")
        if receipt.payment_reference.strip():
            result.append(receipt)
    return tuple(result)


class CalculateTotalTests(unittest.TestCase):
    """Fast unit tests for the pure pricing core."""

    def test_combines_quantities_and_discount(self) -> None:
        items = (
            LineItem("Python workbook", 1_250, 2),
            LineItem("Pen", 150, 1),
        )

        result = calculate_total(items, discount_percent=10)

        self.assertEqual(
            result,
            TotalBreakdown(
                subtotal_cents=2_650,
                discount_cents=265,
                total_cents=2_385,
            ),
        )

    def test_percentage_rounding_is_half_up(self) -> None:
        result = calculate_total([LineItem("Small item", 5)], 10)

        self.assertEqual(result.discount_cents, 1)
        self.assertEqual(result.total_cents, 4)

    def test_discount_boundaries(self) -> None:
        item = LineItem("Course", 2_000)
        cases = ((0, 2_000), (25, 1_500), (100, 0))

        for percent, expected_total in cases:
            with self.subTest(percent=percent):
                actual = calculate_total([item], percent)
                self.assertEqual(actual.total_cents, expected_total)

    def test_empty_cart_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cart cannot be empty"):
            calculate_total([])

    def test_invalid_discount_is_rejected(self) -> None:
        invalid_values = (-1, 101, True)

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    calculate_total([LineItem("Book", 500)], value)  # type: ignore[arg-type]

    def test_line_item_rejects_invalid_quantity(self) -> None:
        for quantity in (0, -1, True):
            with self.subTest(quantity=quantity):
                with self.assertRaises((TypeError, ValueError)):
                    LineItem("Book", 500, quantity)  # type: ignore[arg-type]

    def test_non_line_item_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "LineItem"):
            calculate_total([object()])  # type: ignore[list-item]


class CheckoutServiceTests(unittest.TestCase):
    """Fixture-driven tests for orchestration and boundary behavior."""

    def setUp(self) -> None:
        self.gateway = RecordingGateway()
        self.store = MemoryReceiptStore()
        self.now = datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc)
        self.service = CheckoutService(
            self.gateway,
            self.store,
            id_factory=SequentialId(["order-001"]),
            clock=lambda: self.now,
        )

    def test_success_uses_injected_dependencies(self) -> None:
        receipt = self.service.checkout(
            [LineItem("Testing lesson", 3_000)],
            discount_percent=20,
        )

        self.assertEqual(receipt.order_id, "order-001")
        self.assertEqual(receipt.created_at, self.now)
        self.assertEqual(receipt.total_cents, 2_400)
        self.assertEqual(self.gateway.calls, [("order-001", 2_400)])
        self.assertEqual(self.store.receipts, [receipt])

    def test_gateway_failure_is_translated_and_not_saved(self) -> None:
        gateway = RecordingGateway(error=TimeoutError("offline"))
        service = CheckoutService(
            gateway,
            self.store,
            id_factory=lambda: "order-failed",
            clock=lambda: self.now,
        )

        with self.assertRaisesRegex(CheckoutError, "payment"):
            service.checkout([LineItem("Testing lesson", 3_000)])

        self.assertEqual(gateway.calls, [("order-failed", 3_000)])
        self.assertEqual(self.store.receipts, [])

    def test_naive_clock_is_rejected_before_charging(self) -> None:
        service = CheckoutService(
            self.gateway,
            self.store,
            id_factory=lambda: "order-naive",
            clock=lambda: datetime(2026, 9, 18, 9, 0),
        )

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            service.checkout([LineItem("Testing lesson", 3_000)])

        self.assertEqual(self.gateway.calls, [])


class ReportIntegrationTests(unittest.TestCase):
    """Integration test using an isolated, automatically closed text stream."""

    def setUp(self) -> None:
        self.report = StringIO()
        self.addCleanup(self.report.close)

    def test_report_round_trip_text(self) -> None:
        receipt = Receipt(
            "order-001",
            2_400,
            "pay-test-001",
            datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc),
        )
        expected = (
            "order_id,total_cents,payment_reference,created_at\n"
            "order-001,2400,pay-test-001,2026-09-18T09:00:00+00:00\n"
        )

        characters_written = write_receipt_report(self.report, [receipt])

        self.assertEqual(characters_written, len(expected))
        self.assertEqual(self.report.getvalue(), expected)


class PracticeSolutionTests(unittest.TestCase):
    """Tests that define the exercise solution contracts."""

    def test_shipping_threshold(self) -> None:
        self.assertEqual(
            shipping_fee_cents(
                4_999,
                free_shipping_at_cents=5_000,
                fee_cents=350,
            ),
            350,
        )
        self.assertEqual(
            shipping_fee_cents(
                5_000,
                free_shipping_at_cents=5_000,
                fee_cents=350,
            ),
            0,
        )

    def test_shipping_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            shipping_fee_cents(
                -1,
                free_shipping_at_cents=5_000,
                fee_cents=350,
            )

    def test_sequential_id_exhaustion(self) -> None:
        ids = SequentialId(["order-a", "order-b"])

        self.assertEqual(ids(), "order-a")
        self.assertEqual(ids(), "order-b")
        with self.assertRaisesRegex(RuntimeError, "no prepared IDs"):
            ids()

    def test_successful_receipts_preserve_order(self) -> None:
        now = datetime(2026, 9, 18, tzinfo=timezone.utc)
        first = Receipt("one", 100, "pay-1", now)
        missing = Receipt("two", 200, " ", now)
        third = Receipt("three", 300, "pay-3", now)

        self.assertEqual(
            successful_receipts([first, missing, third]),
            (first, third),
        )


def run_tests() -> unittest.result.TestResult:
    """Discover and run this module's tests with deterministic verbosity."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)


def main() -> None:
    """Run a deterministic checkout demonstration, then the lesson tests."""
    gateway = RecordingGateway("pay-demo-001")
    store = MemoryReceiptStore()
    service = CheckoutService(
        gateway,
        store,
        id_factory=SequentialId(["order-demo-001"]),
        clock=lambda: datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc),
    )
    receipt = service.checkout(
        [
            LineItem("Testing workbook", 1_500, 2),
            LineItem("Reference card", 500),
        ],
        discount_percent=10,
    )

    print("Receipt:", receipt)
    print("Gateway calls:", gateway.calls)
    print("\nRunning lesson tests:")
    result = run_tests()
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
