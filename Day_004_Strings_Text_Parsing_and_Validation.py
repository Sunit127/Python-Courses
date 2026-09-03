"""Day 4: Strings, text parsing, and validation.

Learning goals
--------------
1. Use slicing and string methods without changing the original string.
2. Normalize human-entered text before comparing or storing it.
3. Parse structured text with explicit rules instead of fragile assumptions.
4. Validate text at program boundaries and report useful errors.

Teaching notes
--------------
- Strings are immutable: methods such as strip, replace, and casefold return
  new strings.
- strip removes characters only from the ends; replace can modify matches
  anywhere in a string.
- split separates text into parts, while join combines an iterable of strings.
- casefold is stronger than lower for case-insensitive human text comparisons.
- Use partition when a separator is optional and split when you expect several
  fields. Always validate the resulting number and shape of fields.
- String methods are often clearer than regular expressions. Reach for a
  regular expression when the text pattern, rather than a fixed separator,
  defines the data.
- Validation should happen near input boundaries. Calculation functions can
  then work with clean, trustworthy values.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Write is_palindrome(text), ignoring spaces, punctuation, and letter case.
2. Write mask_card_number(text). Accept spaces and hyphens, require 13 to 19
   digits, and reveal only the final four digits.
3. Write initials(full_name). Collapse extra whitespace and reject empty input.

Solutions appear below the examples.

Expert challenge: configuration parser mini-project
---------------------------------------------------
Build a parser for lines in the form key=value. Ignore blank lines and lines
whose first non-space character is #. Normalize keys to lowercase, reject
invalid or duplicate keys, and keep values as strings. The parse_config
solution below provides a calculation core; a production version could add
typed values, source line numbers, and file input as a separate layer.
"""

import re
from collections.abc import Iterable

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_whitespace(text: str) -> str:
    """Trim text and replace every whitespace run with one ordinary space."""
    return " ".join(text.split())


def same_text(left: str, right: str) -> bool:
    """Compare normalized human text without case distinctions."""
    return normalize_whitespace(left).casefold() == normalize_whitespace(
        right
    ).casefold()


def validate_username(username: str) -> str:
    """Return a cleaned username or raise ValueError.

    A valid username has 3 to 20 characters, starts with a letter, and contains
    only letters, digits, or underscores.
    """
    cleaned = username.strip()
    if not 3 <= len(cleaned) <= 20:
        raise ValueError("username must contain 3 to 20 characters")
    if not cleaned[0].isalpha():
        raise ValueError("username must start with a letter")
    if not all(character.isalnum() or character == "_" for character in cleaned):
        raise ValueError(
            "username may contain only letters, digits, and underscores"
        )
    return cleaned


def parse_inventory_line(line: str) -> tuple[str, str, int, float]:
    """Parse SKU | product name | quantity | unit price."""
    fields = [field.strip() for field in line.split("|")]
    if len(fields) != 4:
        raise ValueError("inventory line must contain exactly four fields")

    sku, product, quantity_text, price_text = fields
    if not sku or not product:
        raise ValueError("SKU and product name cannot be empty")

    try:
        quantity = int(quantity_text)
        unit_price = float(price_text)
    except ValueError as error:
        raise ValueError("quantity and unit price must be numeric") from error

    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if unit_price < 0:
        raise ValueError("unit price cannot be negative")
    return sku, product, quantity, unit_price


def word_frequencies(text: str) -> dict[str, int]:
    """Count ASCII words while preserving first-seen order."""
    counts: dict[str, int] = {}
    for match in WORD_PATTERN.finditer(text):
        word = match.group().casefold()
        counts[word] = counts.get(word, 0) + 1
    return counts


def format_columns(rows: Iterable[tuple[str, str]]) -> list[str]:
    """Align two text columns without trailing whitespace."""
    materialized = [
        (normalize_whitespace(left), normalize_whitespace(right))
        for left, right in rows
    ]
    if not materialized:
        return []
    if any(not left or not right for left, right in materialized):
        raise ValueError("column values cannot be empty")

    left_width = max(len(left) for left, _ in materialized)
    return [f"{left:<{left_width}} | {right}" for left, right in materialized]


# Practice exercise solutions


def is_palindrome(text: str) -> bool:
    """Solution 1: compare only case-folded letters and digits."""
    normalized = "".join(
        character.casefold() for character in text if character.isalnum()
    )
    return normalized == normalized[::-1]


def mask_card_number(text: str) -> str:
    """Solution 2: validate a card-like number and reveal its last four digits.

    This is a formatting exercise, not payment-card validation or secure
    storage. Real applications should use a payment provider and never log
    complete card numbers.
    """
    compact = text.replace(" ", "").replace("-", "")
    if not compact.isdigit() or not 13 <= len(compact) <= 19:
        raise ValueError("card number must contain 13 to 19 digits")
    return "*" * (len(compact) - 4) + compact[-4:]


def initials(full_name: str) -> str:
    """Solution 3: return uppercase initials from a non-empty name."""
    words = normalize_whitespace(full_name).split()
    if not words:
        raise ValueError("full name cannot be empty")
    return "".join(word[0].upper() for word in words)


def parse_config(text: str) -> dict[str, str]:
    """Parse validated key=value settings for the expert challenge."""
    settings: dict[str, str] = {}

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key_text, separator, value_text = line.partition("=")
        if not separator:
            raise ValueError(f"line {line_number}: expected key=value")

        key = key_text.strip().casefold()
        value = value_text.strip()
        if KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"line {line_number}: invalid key {key!r}")
        if not value:
            raise ValueError(f"line {line_number}: value cannot be empty")
        if key in settings:
            raise ValueError(f"line {line_number}: duplicate key {key!r}")

        settings[key] = value

    return settings


def run_self_checks() -> None:
    """Check normal behavior and representative boundary cases."""
    assert normalize_whitespace("  learn\tPython\n daily  ") == "learn Python daily"
    assert same_text("  PYTHON course", "python   COURSE ")
    assert validate_username("  learner_4 ") == "learner_4"
    assert parse_inventory_line("NB-1 | Notebook | 3 | 2.50") == (
        "NB-1",
        "Notebook",
        3,
        2.5,
    )
    assert word_frequencies("Python's syntax; PYTHON'S tools.") == {
        "python's": 2,
        "syntax": 1,
        "tools": 1,
    }
    assert format_columns([]) == []
    assert format_columns([("name", "Asha"), ("score", "91")]) == [
        "name  | Asha",
        "score | 91",
    ]
    assert is_palindrome("Never odd, or even!")
    assert is_palindrome("")
    assert mask_card_number("1234 5678 9012 3456") == "************3456"
    assert initials("  ada   lovelace ") == "AL"
    assert parse_config("# app settings\nHOST = localhost\nport=8000") == {
        "host": "localhost",
        "port": "8000",
    }
    assert parse_config("token=a=b=c") == {"token": "a=b=c"}

    invalid_calls = (
        lambda: validate_username("2bad"),
        lambda: parse_inventory_line("NB-1 | Notebook | 0 | 2.50"),
        lambda: mask_card_number("1234-abcd"),
        lambda: initials("   "),
        lambda: parse_config("port=8000\nPORT=9000"),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input must raise ValueError")


def main() -> None:
    """Run safe, deterministic demonstrations."""
    run_self_checks()

    print("Normalized:", normalize_whitespace("  Daily\tPython practice  "))
    print("Username:", validate_username("learner_4"))
    print(
        "Inventory:",
        parse_inventory_line("PY-101 | Python Workbook | 2 | 14.95"),
    )
    print(
        "Words:",
        word_frequencies("Parse text, validate text, trust the result."),
    )
    print("Palindrome:", is_palindrome("A man, a plan, a canal: Panama"))
    print("Masked number:", mask_card_number("1234-5678-9012-3456"))
    print("Initials:", initials("Grace Brewster Murray Hopper"))

    config_text = """
    # Development settings
    host = localhost
    port = 8000
    debug = false
    """
    print("Configuration:", parse_config(config_text))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
