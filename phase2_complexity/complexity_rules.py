"""
complexity_rules.py — Rule-Based Complexity Detection
=======================================================

Defines a catalogue of algorithmic patterns mapped to their Big-O
complexity classes.  Each rule carries a human-readable description and
a confidence score (0.0–1.0) reflecting how reliably the pattern
implies the stated complexity.

Supported complexity classes
----------------------------
O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2^n), O(n!)

Public API
----------
- ``ComplexityRule``   — dataclass for a single rule
- ``RULES``           — dict mapping pattern names → ``ComplexityRule``
- ``match_rule(name)`` — look up a rule by pattern name
- ``get_all_rules()`` — return every registered rule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplexityRule:
    """Describes a recognised algorithmic pattern and its expected
    time complexity.

    Attributes
    ----------
    pattern_name : str
        Short machine-friendly identifier (e.g. ``"single_loop"``).
    complexity_class : str
        Big-O notation string (e.g. ``"O(n)"``).
    description : str
        Human-readable explanation of the pattern.
    confidence : float
        How reliably the pattern implies the stated complexity.
        Values in [0.0, 1.0].  1.0 = very confident.
    examples : tuple of str
        Short code snippets illustrating the pattern.
    category : str
        Broad category: ``"iterative"``, ``"recursive"``,
        ``"divide_and_conquer"``, ``"lookup"``, ``"sorting"``.
    """

    pattern_name: str
    complexity_class: str
    description: str
    confidence: float = 0.8
    examples: tuple = ()
    category: str = "iterative"


# ---------------------------------------------------------------------------
# Rules catalogue
# ---------------------------------------------------------------------------

RULES: Dict[str, ComplexityRule] = {
    # --- O(1) ---------------------------------------------------------------
    "constant_lookup": ComplexityRule(
        pattern_name="constant_lookup",
        complexity_class="O(1)",
        description=(
            "Direct hash-table / dictionary lookup or array index access. "
            "No loops or recursion involved."
        ),
        confidence=0.95,
        examples=(
            "value = data[key]",
            "return hash_map.get(key, default)",
        ),
        category="lookup",
    ),
    "constant_operation": ComplexityRule(
        pattern_name="constant_operation",
        complexity_class="O(1)",
        description=(
            "A fixed number of arithmetic or logical operations independent "
            "of input size."
        ),
        confidence=0.95,
        examples=("return a + b",),
        category="lookup",
    ),

    # --- O(log n) -----------------------------------------------------------
    "binary_search": ComplexityRule(
        pattern_name="binary_search",
        complexity_class="O(log n)",
        description=(
            "Binary search pattern: repeatedly halves the search space "
            "using a midpoint calculation (mid = (lo + hi) // 2)."
        ),
        confidence=0.90,
        examples=(
            "while lo <= hi:\n    mid = (lo + hi) // 2\n    if arr[mid] == target: ...",
        ),
        category="divide_and_conquer",
    ),
    "logarithmic_loop": ComplexityRule(
        pattern_name="logarithmic_loop",
        complexity_class="O(log n)",
        description=(
            "A loop whose counter is multiplied or divided by a constant "
            "each iteration (e.g. i *= 2 or i //= 2)."
        ),
        confidence=0.85,
        examples=(
            "i = n\nwhile i > 0:\n    i //= 2",
        ),
        category="iterative",
    ),

    # --- O(n) ---------------------------------------------------------------
    "single_loop": ComplexityRule(
        pattern_name="single_loop",
        complexity_class="O(n)",
        description=(
            "A single loop iterating over the input once (e.g. a for-each "
            "loop or a while loop with linear progression)."
        ),
        confidence=0.90,
        examples=(
            "for item in collection:\n    process(item)",
            "i = 0\nwhile i < n:\n    i += 1",
        ),
        category="iterative",
    ),
    "linear_scan": ComplexityRule(
        pattern_name="linear_scan",
        complexity_class="O(n)",
        description=(
            "Scanning an entire list or array once to find an element, "
            "compute a sum, or apply a transformation."
        ),
        confidence=0.90,
        examples=(
            "total = sum(arr)",
            "result = [f(x) for x in arr]",
        ),
        category="iterative",
    ),
    "linear_recursion": ComplexityRule(
        pattern_name="linear_recursion",
        complexity_class="O(n)",
        description=(
            "Tail recursion or simple recursion that decrements the problem "
            "size by 1 each call (e.g. factorial)."
        ),
        confidence=0.80,
        examples=(
            "def fact(n): return 1 if n <= 1 else n * fact(n-1)",
        ),
        category="recursive",
    ),

    # --- O(n log n) ---------------------------------------------------------
    "merge_sort": ComplexityRule(
        pattern_name="merge_sort",
        complexity_class="O(n log n)",
        description=(
            "Divide-and-conquer sorting: split input in half, recursively "
            "sort both halves, then merge in O(n)."
        ),
        confidence=0.85,
        examples=(
            "def merge_sort(arr):\n    mid = len(arr)//2\n    left = merge_sort(arr[:mid])\n    ...",
        ),
        category="sorting",
    ),
    "divide_and_conquer": ComplexityRule(
        pattern_name="divide_and_conquer",
        complexity_class="O(n log n)",
        description=(
            "General divide-and-conquer pattern: split the problem, solve "
            "sub-problems recursively, combine results."
        ),
        confidence=0.75,
        examples=(
            "def solve(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return combine(solve(arr[:mid]), solve(arr[mid:]))",
        ),
        category="divide_and_conquer",
    ),
    "efficient_sort": ComplexityRule(
        pattern_name="efficient_sort",
        complexity_class="O(n log n)",
        description=(
            "Built-in sort or heapsort-style algorithm with O(n log n) "
            "average-case guarantee."
        ),
        confidence=0.90,
        examples=(
            "arr.sort()",
            "sorted(arr)",
        ),
        category="sorting",
    ),

    # --- O(n²) --------------------------------------------------------------
    "nested_loop_2": ComplexityRule(
        pattern_name="nested_loop_2",
        complexity_class="O(n²)",
        description=(
            "Two nested loops each iterating up to n times, giving "
            "quadratic time."
        ),
        confidence=0.90,
        examples=(
            "for i in range(n):\n    for j in range(n):\n        ...",
        ),
        category="iterative",
    ),
    "bubble_sort": ComplexityRule(
        pattern_name="bubble_sort",
        complexity_class="O(n²)",
        description=(
            "Bubble-sort or insertion-sort pattern: nested passes over "
            "the array with adjacent swaps."
        ),
        confidence=0.90,
        examples=(
            "for i in range(n):\n    for j in range(n-i-1):\n        if arr[j]>arr[j+1]: swap",
        ),
        category="sorting",
    ),
    "selection_sort": ComplexityRule(
        pattern_name="selection_sort",
        complexity_class="O(n²)",
        description=(
            "Selection-sort pattern: find the minimum in the unsorted "
            "portion and swap it into place."
        ),
        confidence=0.85,
        examples=(
            "for i in range(n):\n    min_idx = i\n    for j in range(i+1, n):\n        if arr[j] < arr[min_idx]: min_idx = j\n    swap(i, min_idx)",
        ),
        category="sorting",
    ),

    # --- O(n³) --------------------------------------------------------------
    "nested_loop_3": ComplexityRule(
        pattern_name="nested_loop_3",
        complexity_class="O(n³)",
        description=(
            "Three nested loops each iterating up to n times (e.g. "
            "matrix multiplication)."
        ),
        confidence=0.90,
        examples=(
            "for i in range(n):\n    for j in range(n):\n        for k in range(n):\n            C[i][j] += A[i][k]*B[k][j]",
        ),
        category="iterative",
    ),

    # --- O(2^n) -------------------------------------------------------------
    "exponential_recursion": ComplexityRule(
        pattern_name="exponential_recursion",
        complexity_class="O(2^n)",
        description=(
            "Recursive function that branches into two (or more) "
            "sub-problems of size n-1 without memoisation (e.g. naïve "
            "Fibonacci)."
        ),
        confidence=0.85,
        examples=(
            "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
        ),
        category="recursive",
    ),
    "power_set": ComplexityRule(
        pattern_name="power_set",
        complexity_class="O(2^n)",
        description=(
            "Generating all subsets of a set — 2^n possible subsets."
        ),
        confidence=0.80,
        examples=(
            "def subsets(s):\n    if not s: return [[]]\n    rest = subsets(s[1:])\n    return rest + [[s[0]]+r for r in rest]",
        ),
        category="recursive",
    ),

    # --- O(n!) --------------------------------------------------------------
    "permutations": ComplexityRule(
        pattern_name="permutations",
        complexity_class="O(n!)",
        description=(
            "Generating all permutations of n elements — n! total "
            "arrangements."
        ),
        confidence=0.80,
        examples=(
            "def perms(arr):\n    if len(arr)<=1: return [arr]\n    result=[]\n    for i,v in enumerate(arr):\n        for p in perms(arr[:i]+arr[i+1:]):\n            result.append([v]+p)\n    return result",
        ),
        category="recursive",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def match_rule(pattern_name: str) -> ComplexityRule:
    """Look up a complexity rule by its pattern name.

    Parameters
    ----------
    pattern_name : str
        The ``pattern_name`` field of the rule to retrieve (case-insensitive).

    Returns
    -------
    ComplexityRule
        The matching rule.

    Raises
    ------
    KeyError
        If no rule with the given name exists.
    """
    key = pattern_name.lower().strip()
    if key in RULES:
        return RULES[key]

    # Fuzzy fallback: try partial match
    candidates = [name for name in RULES if key in name or name in key]
    if len(candidates) == 1:
        logger.debug("Fuzzy-matched rule '%s' → '%s'", pattern_name, candidates[0])
        return RULES[candidates[0]]

    raise KeyError(
        f"No complexity rule named '{pattern_name}'. "
        f"Available rules: {sorted(RULES.keys())}"
    )


def get_all_rules() -> List[ComplexityRule]:
    """Return a list of every registered complexity rule.

    Returns
    -------
    List[ComplexityRule]
        All rules sorted by complexity class then pattern name.
    """
    _ORDER = {
        "O(1)": 0, "O(log n)": 1, "O(n)": 2, "O(n log n)": 3,
        "O(n²)": 4, "O(n³)": 5, "O(2^n)": 6, "O(n!)": 7,
    }

    return sorted(
        RULES.values(),
        key=lambda r: (_ORDER.get(r.complexity_class, 99), r.pattern_name),
    )


def get_rules_by_category(category: str) -> List[ComplexityRule]:
    """Return all rules belonging to *category*.

    Parameters
    ----------
    category : str
        E.g. ``"iterative"``, ``"recursive"``, ``"sorting"``.

    Returns
    -------
    List[ComplexityRule]
    """
    cat = category.lower().strip()
    return [r for r in RULES.values() if r.category == cat]


def get_rules_by_complexity(complexity_class: str) -> List[ComplexityRule]:
    """Return all rules with the given Big-O class.

    Parameters
    ----------
    complexity_class : str
        E.g. ``"O(n)"``, ``"O(n²)"``.

    Returns
    -------
    List[ComplexityRule]
    """
    cc = complexity_class.strip()
    return [r for r in RULES.values() if r.complexity_class == cc]


def summarise_rules() -> str:
    """Return a human-readable summary table of all registered rules.

    Returns
    -------
    str
        Formatted multi-line string.
    """
    lines: List[str] = []
    lines.append(f"{'Pattern':<28} {'Complexity':<12} {'Confidence':>10}  Description")
    lines.append("-" * 90)
    for rule in get_all_rules():
        conf_str = f"{rule.confidence:.0%}"
        desc_short = rule.description[:40] + ("…" if len(rule.description) > 40 else "")
        lines.append(
            f"{rule.pattern_name:<28} {rule.complexity_class:<12} {conf_str:>10}  {desc_short}"
        )
    return "\n".join(lines)
