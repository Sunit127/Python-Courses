"""Day 3: Loops, collections, and reusable data processing.

Learning goals
--------------
1. Choose a list, tuple, set, or dictionary for the job.
2. Traverse data with ``for``, ``range``, ``enumerate``, and ``zip``.
3. Build new collections with comprehensions.
4. Accumulate reliable summaries while validating input.

Teaching notes
--------------
- A list is ordered and mutable; a tuple is ordered and fixed.
- A set stores unique hashable values and supports fast membership checks.
- A dictionary maps unique keys to values.
- Prefer direct iteration (``for value in values``). Use ``enumerate`` when
  you also need an index, and ``zip`` when walking collections together.
- Comprehensions are ideal for small transformations. Use a regular loop
  when validation or several steps would make a comprehension hard to read.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Build a frequency table for any list of strings.
2. Flatten a rectangular or ragged list of lists.
3. Group names by their first letter without losing input order.

Solutions appear below the examples.

Expert challenge: sales summary mini-project
--------------------------------------------
Extend ``build_sales_report`` so it also returns the best-selling category by
revenue. Then write an input layer that gathers sales without changing the
report function. Keep invalid quantities and prices out of the report.
"""

from string import punctuation

Sale = tuple[str, str, int, float]


def clean_words(text: str) -> list[str]:
    """Return lowercase words with ASCII punctuation removed."""
    translation_table = str.maketrans("", "", punctuation)
    return text.lower().translate(translation_table).split()


def word_frequencies(text: str) -> dict[str, int]:
    """Count each cleaned word in insertion order."""
    frequencies: dict[str, int] = {}
    for word in clean_words(text):
        frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


def numbered_tasks(tasks: list[str]) -> list[str]:
    """Format non-empty task names with one-based positions."""
    result: list[str] = []
    for position, task in enumerate(tasks, start=1):
        cleaned_task = task.strip()
        if not cleaned_task:
            raise ValueError("task names cannot be empty")
        result.append(f"{position}. {cleaned_task}")
    return result


def combine_names_and_scores(
    names: list[str],
    scores: list[float],
) -> list[tuple[str, float]]:
    """Pair equally sized name and score lists after validation."""
    if len(names) != len(scores):
        raise ValueError("names and scores must have equal lengths")

    combined: list[tuple[str, float]] = []
    for name, score in zip(names, scores):
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("student names cannot be empty")
        if not 0 <= score <= 100:
            raise ValueError("scores must be between 0 and 100")
        combined.append((cleaned_name, score))
    return combined


def inventory_totals(shipments: list[tuple[str, int]]) -> dict[str, int]:
    """Accumulate positive shipment quantities by product."""
    totals: dict[str, int] = {}
    for product, quantity in shipments:
        cleaned_product = product.strip()
        if not cleaned_product:
            raise ValueError("product names cannot be empty")
        if quantity <= 0:
            raise ValueError("shipment quantities must be positive")
        totals[cleaned_product] = totals.get(cleaned_product, 0) + quantity
    return totals


def moving_averages(values: list[float], window: int) -> list[float]:
    """Return averages for each complete sliding window."""
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(values):
        return []

    averages: list[float] = []
    for start in range(len(values) - window + 1):
        current_window = values[start : start + window]
        averages.append(round(sum(current_window) / window, 2))
    return averages


# Practice exercise solutions


def frequency_table(items: list[str]) -> dict[str, int]:
    """Solution 1: count strings without sorting or discarding order."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def flatten(matrix: list[list[int]]) -> list[int]:
    """Solution 2: flatten rows of any length."""
    return [value for row in matrix for value in row]


def group_names(names: list[str]) -> dict[str, list[str]]:
    """Solution 3: group cleaned names by uppercase first letter."""
    groups: dict[str, list[str]] = {}
    for name in names:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("names cannot be empty")
        first_letter = cleaned_name[0].upper()
        groups.setdefault(first_letter, []).append(cleaned_name)
    return groups


def build_sales_report(sales: list[Sale]) -> dict[str, object]:
    """Build the calculation core for the expert mini-project.

    Each sale is ``(product, category, units, unit_price)``. An empty sales
    list is valid and produces a zero-valued report.
    """
    product_revenue: dict[str, float] = {}
    category_revenue: dict[str, float] = {}
    total_units = 0

    for product, category, units, unit_price in sales:
        cleaned_product = product.strip()
        cleaned_category = category.strip()
        if not cleaned_product or not cleaned_category:
            raise ValueError("product and category names cannot be empty")
        if units <= 0:
            raise ValueError("units must be positive")
        if unit_price < 0:
            raise ValueError("unit_price cannot be negative")

        revenue = units * unit_price
        total_units += units
        product_revenue[cleaned_product] = (
            product_revenue.get(cleaned_product, 0.0) + revenue
        )
        category_revenue[cleaned_category] = (
            category_revenue.get(cleaned_category, 0.0) + revenue
        )

    top_product = (
        max(product_revenue, key=product_revenue.get)
        if product_revenue
        else None
    )
    total_revenue = sum(product_revenue.values())

    return {
        "total_units": total_units,
        "total_revenue": round(total_revenue, 2),
        "unique_products": len(product_revenue),
        "top_product": top_product,
        "category_revenue": {
            category: round(revenue, 2)
            for category, revenue in category_revenue.items()
        },
    }


def run_self_checks() -> None:
    """Check representative results and important edge cases."""
    assert word_frequencies("Python, Python! Loops.") == {
        "python": 2,
        "loops": 1,
    }
    assert numbered_tasks([]) == []
    assert combine_names_and_scores(["Asha"], [91]) == [("Asha", 91)]
    assert inventory_totals([("pen", 2), ("pen", 3)]) == {"pen": 5}
    assert moving_averages([2, 4, 8, 10], 2) == [3.0, 6.0, 9.0]
    assert moving_averages([1, 2], 3) == []
    assert flatten([[1, 2], [], [3]]) == [1, 2, 3]
    assert build_sales_report([])["top_product"] is None

    try:
        combine_names_and_scores(["Asha"], [101])
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range score must be rejected")


def main() -> None:
    """Run safe, self-contained demonstrations."""
    run_self_checks()

    print("Word counts:", word_frequencies("Loops make data useful; loops repeat."))
    print("Tasks:", numbered_tasks(["read notes", "solve exercises"]))
    print(
        "Students:",
        combine_names_and_scores(["Asha", "Bikash"], [91, 84]),
    )
    print(
        "Inventory:",
        inventory_totals([("notebook", 4), ("pen", 10), ("notebook", 2)]),
    )
    print("Moving averages:", moving_averages([10, 15, 20, 30], window=3))

    sample_sales: list[Sale] = [
        ("Keyboard", "Accessories", 2, 45.0),
        ("Mouse", "Accessories", 3, 20.0),
        ("Keyboard", "Accessories", 1, 45.0),
        ("Course", "Education", 4, 25.0),
    ]
    print("Sales report:", build_sales_report(sample_sales))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
