"""Day 5: Exceptions, defensive programming, and reliable error handling.

Learning goals
--------------
1. Distinguish expected input failures from programming bugs.
2. Catch specific exceptions at the boundary where you can recover.
3. Raise clear, domain-specific exceptions from validation functions.
4. Use try/except/else/finally deliberately and preserve error context.
5. Process unreliable records without hiding failures.

Teaching notes
--------------
- An exception is a signal that normal execution cannot continue in its
  current path. Handle it where you have a useful recovery or user message.
- Catch the narrowest exception you expect. A bare "except:" can hide
  KeyboardInterrupt, programming mistakes, and operational failures.
- Validate at boundaries, then keep the core calculation simple.
- Use "raise ValueError(...)" for a bad argument and a custom exception when
  callers need to distinguish one business failure from another.
- The else block runs only when the try block succeeds; finally runs whether
  it succeeds or fails and is best reserved for cleanup.
- "raise NewError(...) from error" keeps the original cause while presenting a
  domain-level message. Never print a password, token, or complete payment
  number while reporting an error.
- For a batch, catch a known record error per item so one bad row does not
  discard valid rows. Do not catch every exception: unexpected bugs should
  still fail loudly during development.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Write safe_int(text, default) that returns an integer or the supplied
   default when text is not a valid integer. Do not catch unrelated errors.
2. Write bounded_percent(text) that parses a number and accepts only 0 through
   100, raising ValidationError with a helpful message otherwise.
3. Write first_valid_amount(values) that returns the first non-negative float
   and raises LookupError("no valid amount") when none exists.

Solutions appear below the examples.

Expert challenge: resilient transaction import
----------------------------------------------
Extend process_transactions so it accepts a configurable decimal precision and
returns a sorted list of the highest-value accepted transactions. Add a
duplicate transaction-id check without changing parse_transaction_line.
Then design a command-line or file-input layer that reports rejected line
numbers while keeping the calculation core free of input and output calls.
"""

from collections.abc import Iterable

Transaction = tuple[str, str, float]


class CourseError(Exception):
    """Base class for expected errors in this lesson."""


class ValidationError(CourseError):
    """A supplied value has the right shape but an invalid value."""


class RecordFormatError(CourseError):
    """A record cannot be parsed into the expected fields."""


def divide_text(numerator_text: str, denominator_text: str) -> float:
    """Parse two numbers and divide them, demonstrating try/except/else.

    Conversion failures and division by zero become one clear public error.
    The original exception is retained as the cause for debugging.
    """
    try:
        numerator = float(numerator_text.strip())
        denominator = float(denominator_text.strip())
    except (AttributeError, ValueError) as error:
        raise ValidationError("both values must be numeric text") from error
    else:
        if denominator == 0:
            raise ValidationError("denominator cannot be zero")
        return numerator / denominator


def parse_non_negative_amount(text: str) -> float:
    """Return a finite, non-negative amount or raise ValidationError."""
    try:
        amount = float(text.strip())
    except (AttributeError, ValueError) as error:
        raise ValidationError("amount must be numeric text") from error

    if amount < 0:
        raise ValidationError("amount cannot be negative")
    if amount != amount or amount in (float("inf"), float("-inf")):
        raise ValidationError("amount must be finite")
    return amount


def parse_transaction_line(line: str) -> Transaction:
    """Parse transaction_id | category | amount with explicit errors."""
    fields = [field.strip() for field in line.split("|")]
    if len(fields) != 3:
        raise RecordFormatError(
            "transaction must contain id, category, and amount"
        )

    transaction_id, category, amount_text = fields
    if not transaction_id or not category:
        raise RecordFormatError("transaction id and category are required")

    try:
        amount = parse_non_negative_amount(amount_text)
    except ValidationError as error:
        raise RecordFormatError(
            f"invalid amount for transaction {transaction_id!r}"
        ) from error

    return transaction_id, category, amount


def process_transactions(
    lines: Iterable[str],
) -> dict[str, object]:
    """Import valid transactions while collecting expected record failures.

    The function catches only CourseError. A programming error such as a
    misspelled variable should not be converted into a misleading rejection.
    """
    accepted: list[Transaction] = []
    rejected: list[dict[str, object]] = []
    category_totals: dict[str, float] = {}

    for line_number, line in enumerate(lines, start=1):
        try:
            transaction = parse_transaction_line(line)
        except CourseError as error:
            rejected.append(
                {"line": line_number, "error": str(error)}
            )
            continue

        accepted.append(transaction)
        _, category, amount = transaction
        category_totals[category] = category_totals.get(category, 0.0) + amount

    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "category_totals": {
            category: round(total, 2)
            for category, total in category_totals.items()
        },
    }


def safe_int(text: str, default: int) -> int:
    """Solution 1: use a narrow ValueError recovery path."""
    try:
        return int(text.strip())
    except (AttributeError, ValueError):
        return default


def bounded_percent(text: str) -> float:
    """Solution 2: parse a number, then enforce its domain."""
    try:
        percent = float(text.strip())
    except (AttributeError, ValueError) as error:
        raise ValidationError("percent must be numeric text") from error

    if percent != percent or percent in (float("inf"), float("-inf")):
        raise ValidationError("percent must be finite")
    if not 0 <= percent <= 100:
        raise ValidationError("percent must be between 0 and 100")
    return percent


def first_valid_amount(values: Iterable[object]) -> float:
    """Solution 3: skip malformed values and find the first valid amount."""
    for value in values:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if amount >= 0 and amount not in (float("inf"), float("-inf")):
            return amount
    raise LookupError("no valid amount")


def run_self_checks() -> None:
    """Check normal behavior, error causes, and important edge cases."""
    assert divide_text(" 9 ", "2") == 4.5
    assert parse_non_negative_amount("0") == 0.0
    assert safe_int(" 42 ", -1) == 42
    assert safe_int("not a number", -1) == -1
    assert bounded_percent("12.5") == 12.5
    assert first_valid_amount(["bad", -3, "4.25"]) == 4.25

    report = process_transactions(
        [
            "T-1 | books | 12.50",
            "broken row",
            "T-2 | tools | 5",
            "T-3 | books | -1",
            "T-4 | tools | 2.25",
        ]
    )
    assert report["accepted_count"] == 3
    assert report["rejected_count"] == 2
    assert report["category_totals"] == {"books": 12.5, "tools": 7.25}
    assert report["rejected"][0]["line"] == 2

    try:
        divide_text("not-a-number", "2")
    except ValidationError as error:
        assert isinstance(error.__cause__, ValueError)
    else:
        raise AssertionError("bad numeric text must preserve its cause")

    try:
        divide_text("10", "0")
    except ValidationError as error:
        assert str(error) == "denominator cannot be zero"
    else:
        raise AssertionError("division by zero must be rejected")

    invalid_calls = (
        lambda: parse_transaction_line("T-1 | books"),
        lambda: parse_transaction_line("T-2 | tools | nope"),
        lambda: bounded_percent("101"),
        lambda: bounded_percent("nan"),
        lambda: first_valid_amount(["x", -1]),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (CourseError, LookupError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without user input."""
    run_self_checks()

    print("Division:", divide_text("18", "3"))
    print("Fallback integer:", safe_int("not-an-int", 0))
    print("Percent:", bounded_percent("87.5"))

    sample_lines = [
        "TX-100 | books | 19.95",
        "TX-101 | tools | 8.50",
        "TX-102 | books | 12.00",
        "TX-103 | tools | -2.00",
    ]
    print("Transaction report:", process_transactions(sample_lines))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
