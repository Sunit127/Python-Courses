"""Day 2: Functions and structured problem solving.

Learning goals
--------------
1. Define reusable functions with parameters and return values.
2. Validate inputs before doing calculations.
3. Break a problem into small, testable steps.
4. Use built-in functions instead of duplicating logic.

Run this file with Python 3 to see the examples at the bottom.

Practice exercises
------------------
1. Write ``is_even(number)`` and return a Boolean.
2. Write ``count_vowels(text)`` and ignore letter case.
3. Extend ``shopping_summary`` so it accepts a delivery fee.

Expert challenge
----------------
Build a command-line expense tracker. Ask for several expenses, reject
negative values, and print their total, average, smallest, and largest values.
Start by writing one function for input and another for the summary.
"""


def greet(name: str) -> str:
    """Return a friendly greeting for a non-empty name."""
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("name cannot be empty")
    return f"Hello, {cleaned_name}!"


def classify_score(score: float) -> str:
    """Convert a score from 0 to 100 into a grade."""
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")

    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    return "Needs improvement"


def safe_divide(dividend: float, divisor: float) -> float | None:
    """Return the quotient, or None when division is impossible."""
    if divisor == 0:
        return None
    return dividend / divisor


def largest_value(*numbers: float) -> float:
    """Return the largest supplied number and handle ties correctly."""
    if not numbers:
        raise ValueError("provide at least one number")
    return max(numbers)


def shopping_summary(
    prices: list[float],
    tax_rate: float = 0.13,
    discount_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate a shopping subtotal, discount, tax, and final total.

    Rates are decimal fractions: 13% is written as ``0.13``.
    """
    if any(price < 0 for price in prices):
        raise ValueError("prices cannot be negative")
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate must be between 0 and 1")
    if not 0 <= discount_rate <= 1:
        raise ValueError("discount_rate must be between 0 and 1")

    subtotal = sum(prices)
    discount = subtotal * discount_rate
    taxable_amount = subtotal - discount
    tax = taxable_amount * tax_rate
    total = taxable_amount + tax

    return {
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
    }


def is_even(number: int) -> bool:
    """Solution to practice exercise 1."""
    return number % 2 == 0


def count_vowels(text: str) -> int:
    """Solution to practice exercise 2."""
    vowels = {"a", "e", "i", "o", "u"}
    return sum(character.lower() in vowels for character in text)


def expense_summary(expenses: list[float]) -> dict[str, float]:
    """Core calculation for the expert challenge."""
    if not expenses:
        raise ValueError("provide at least one expense")
    if any(expense < 0 for expense in expenses):
        raise ValueError("expenses cannot be negative")

    return {
        "total": round(sum(expenses), 2),
        "average": round(sum(expenses) / len(expenses), 2),
        "smallest": min(expenses),
        "largest": max(expenses),
    }


def main() -> None:
    """Run a small demonstration without requiring user input."""
    print(greet("Sunit"))
    print("Grade for 86:", classify_score(86))

    quotient = safe_divide(10, 0)
    print("Division result:", "Cannot divide by zero" if quotient is None else quotient)

    print("Largest value:", largest_value(10, 10, 4))
    print("Shopping summary:", shopping_summary([1200.0, 850.0, 400.0], discount_rate=0.10))
    print("Expense summary:", expense_summary([500.0, 1200.0, 300.0]))
    print("Vowels in Python:", count_vowels("Python"))


if __name__ == "__main__":
    main()
