"""
loop_analyzer.py — Loop Structure & Complexity Estimator
==========================================================

Analyses the loop structures, recursion patterns, and algorithmic idioms
inside parsed Python ASTs to produce per-function Big-O complexity
estimates with confidence scores.

Data Classes
------------
- ``LoopNestInfo``       — information about one loop nesting chain
- ``ComplexityEstimate`` — estimated Big-O for a single function

Public API
----------
- ``analyze_loop_nesting(node)``
- ``detect_recursive_calls(node, func_name)``
- ``detect_binary_search_pattern(node)``
- ``detect_divide_and_conquer(node)``
- ``estimate_function_complexity(func_node, func_name)``
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from phase2_complexity.complexity_rules import ComplexityRule, match_rule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoopNestInfo:
    """Describes a single chain of nested loops inside a function.

    Attributes
    ----------
    function_name : str
        Enclosing function (or ``"<module>"`` for top-level code).
    depth : int
        Maximum nesting depth of the chain (1 = single loop).
    loop_types : list of str
        Ordered list of loop keywords from outermost to innermost
        (e.g. ``["for", "for"]`` for a doubly-nested for loop).
    start_line : int
        Line number of the outermost loop.
    end_line : int
        Line number of the end of the innermost loop body.
    has_break : bool
        Whether any loop in the chain contains a ``break`` statement.
    has_early_return : bool
        Whether any loop body contains a ``return`` statement.
    """

    function_name: str = "<module>"
    depth: int = 1
    loop_types: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    has_break: bool = False
    has_early_return: bool = False


@dataclass
class ComplexityEstimate:
    """Estimated time complexity for a single function.

    Attributes
    ----------
    function_name : str
    complexity_class : str
        Big-O string (e.g. ``"O(n²)"``).
    confidence : float
        0.0–1.0 confidence in the estimate.
    matched_pattern : str
        Name of the matched ``ComplexityRule`` pattern.
    explanation : str
        Human-readable rationale.
    loop_nests : list of LoopNestInfo
        All loop nesting chains found.
    is_recursive : bool
    start_line : int
    end_line : int
    """

    function_name: str = ""
    complexity_class: str = "O(1)"
    confidence: float = 0.5
    matched_pattern: str = "constant_operation"
    explanation: str = ""
    loop_nests: List[LoopNestInfo] = field(default_factory=list)
    is_recursive: bool = False
    start_line: int = 0
    end_line: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_loop(node: ast.AST) -> bool:
    """Return ``True`` if *node* is a ``for`` or ``while`` loop."""
    return isinstance(node, (ast.For, ast.While))


def _loop_keyword(node: ast.AST) -> str:
    if isinstance(node, ast.For):
        return "for"
    if isinstance(node, ast.While):
        return "while"
    return "unknown"


def _body_contains(body: List[ast.stmt], node_type: type) -> bool:
    """Check whether *body* (a list of statements) contains any node of
    *node_type* at any depth.
    """
    for stmt in body:
        for child in ast.walk(stmt):
            if isinstance(child, node_type):
                return True
    return False


def _contains_call_to(node: ast.AST, name: str) -> bool:
    """Return ``True`` if *node* contains a ``Call`` to *name*."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id == name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == name:
                return True
    return False


# ---------------------------------------------------------------------------
# Loop nesting analysis
# ---------------------------------------------------------------------------

def analyze_loop_nesting(node: ast.AST) -> List[LoopNestInfo]:
    """Find every loop-nesting chain under *node*.

    Each chain represents a path from an outermost loop down to its
    deepest nested loop.  Non-nested loops are returned as depth-1 chains.

    Parameters
    ----------
    node : ast.AST
        The AST root (typically an ``ast.Module`` or ``ast.FunctionDef``).

    Returns
    -------
    List[LoopNestInfo]
        One entry per outermost loop encountered, recording the maximum
        depth reached within that loop.
    """
    results: List[LoopNestInfo] = []

    # Determine the enclosing function name
    func_name = "<module>"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        func_name = node.name

    def _collect_chains(
        current: ast.AST,
        chain: List[str],
        outermost_line: int,
    ) -> None:
        for child in ast.iter_child_nodes(current):
            if _is_loop(child):
                new_chain = chain + [_loop_keyword(child)]
                start = outermost_line if outermost_line else child.lineno
                end = child.end_lineno or child.lineno

                body = child.body if hasattr(child, "body") else []
                has_brk = _body_contains(body, ast.Break)
                has_ret = _body_contains(body, ast.Return)

                # Check for deeper nesting inside this loop's body
                has_deeper = False
                for stmt in body:
                    for desc in ast.walk(stmt):
                        if _is_loop(desc) and desc is not child:
                            has_deeper = True
                            break
                    if has_deeper:
                        break

                if not has_deeper:
                    # Leaf of nesting chain — record it
                    results.append(LoopNestInfo(
                        function_name=func_name,
                        depth=len(new_chain),
                        loop_types=new_chain,
                        start_line=start,
                        end_line=end,
                        has_break=has_brk,
                        has_early_return=has_ret,
                    ))
                else:
                    # Recurse into body
                    _collect_chains(child, new_chain, start)
            else:
                # Non-loop node — keep searching
                _collect_chains(child, chain, outermost_line)

    _collect_chains(node, [], 0)
    return results


# ---------------------------------------------------------------------------
# Recursion detection
# ---------------------------------------------------------------------------

def detect_recursive_calls(node: ast.AST, func_name: str) -> bool:
    """Detect whether *node* (a function body) calls *func_name* recursively.

    Parameters
    ----------
    node : ast.AST
        Typically an ``ast.FunctionDef`` or its body.
    func_name : str
        The name of the enclosing function.

    Returns
    -------
    bool
        ``True`` if a recursive call is found.
    """
    if not func_name:
        return False
    return _contains_call_to(node, func_name)


def _count_recursive_branches(node: ast.AST, func_name: str) -> int:
    """Count the number of direct recursive calls in the function body.

    This helps distinguish linear recursion (1 call) from tree recursion
    (2+ calls, e.g. Fibonacci).
    """
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id == func_name:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Binary search detection
# ---------------------------------------------------------------------------

def detect_binary_search_pattern(node: ast.AST) -> bool:
    """Detect a binary-search-like halving pattern.

    Looks for a ``while`` loop containing a midpoint calculation of the
    form ``mid = (lo + hi) // 2`` (or variations) and conditional
    adjustment of bounds.

    Parameters
    ----------
    node : ast.AST
        Function or module AST.

    Returns
    -------
    bool
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.While):
            continue

        body = child.body
        has_mid_calc = False
        has_bound_adjust = False

        for stmt in body:
            for desc in ast.walk(stmt):
                # Look for mid = (lo + hi) // 2 or mid = (lo + hi) / 2
                if isinstance(desc, ast.Assign):
                    for target in desc.targets:
                        if isinstance(target, ast.Name) and target.id in ("mid", "middle", "m"):
                            has_mid_calc = True
                            break
                # Look for lo = mid + 1 or hi = mid - 1  (bound adjustments)
                if isinstance(desc, ast.Assign):
                    if isinstance(desc.value, ast.BinOp):
                        if isinstance(desc.value.op, (ast.Add, ast.Sub)):
                            for target in desc.targets:
                                if isinstance(target, ast.Name) and target.id in (
                                    "lo", "low", "left", "l", "start",
                                    "hi", "high", "right", "r", "end",
                                ):
                                    has_bound_adjust = True

        if has_mid_calc and has_bound_adjust:
            return True

    # Also check for recursive binary search style
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check for mid calculation + recursive call + reduced range
            body_src_has_mid = False
            for desc in ast.walk(child):
                if isinstance(desc, ast.Assign):
                    for target in desc.targets:
                        if isinstance(target, ast.Name) and target.id in ("mid", "middle", "m"):
                            body_src_has_mid = True
            if body_src_has_mid and _contains_call_to(child, child.name):
                return True

    return False


# ---------------------------------------------------------------------------
# Divide-and-conquer detection
# ---------------------------------------------------------------------------

def detect_divide_and_conquer(node: ast.AST) -> bool:
    """Detect a divide-and-conquer pattern.

    Looks for a recursive function that:
    1. Has a base case (``if len(…) <= 1`` or similar).
    2. Splits the input (slicing or midpoint).
    3. Makes 2+ recursive calls on sub-problems.

    Parameters
    ----------
    node : ast.AST

    Returns
    -------
    bool
    """
    for child in ast.walk(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = child.name
        recursive_count = _count_recursive_branches(child, func_name)
        if recursive_count < 2:
            continue

        # Check for input splitting (slicing)
        has_split = False
        for desc in ast.walk(child):
            if isinstance(desc, ast.Subscript):
                if isinstance(desc.slice, ast.Slice):
                    has_split = True
                    break
            # Also check for midpoint variable
            if isinstance(desc, ast.Assign):
                for target in desc.targets:
                    if isinstance(target, ast.Name) and target.id in ("mid", "middle", "half", "pivot"):
                        has_split = True
                        break

        # Check for a base case
        has_base_case = False
        for desc in ast.walk(child):
            if isinstance(desc, ast.If):
                # Check if the test involves len() or a comparison with a small constant
                test = desc.test
                if isinstance(test, ast.Compare):
                    has_base_case = True
                    break
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    has_base_case = True
                    break

        if has_split and has_base_case:
            return True

    return False


# ---------------------------------------------------------------------------
# Logarithmic loop detection
# ---------------------------------------------------------------------------

def _detect_logarithmic_loop(node: ast.AST) -> bool:
    """Detect a loop whose counter doubles or halves each iteration.

    Patterns:
    - ``i *= 2``, ``i //= 2``, ``i /= 2``, ``i >>= 1``
    - ``i = i * 2``, ``i = i // 2``
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.While):
            continue

        for stmt in child.body:
            for desc in ast.walk(stmt):
                # AugAssign: i *= 2, i //= 2, i >>= 1
                if isinstance(desc, ast.AugAssign):
                    if isinstance(desc.op, (ast.Mult, ast.FloorDiv, ast.Div, ast.RShift)):
                        if isinstance(desc.value, ast.Constant) and desc.value.value in (2, 1):
                            return True
                # Assign: i = i * 2, i = i // 2
                if isinstance(desc, ast.Assign) and isinstance(desc.value, ast.BinOp):
                    if isinstance(desc.value.op, (ast.Mult, ast.FloorDiv, ast.Div)):
                        if isinstance(desc.value.right, ast.Constant) and desc.value.right.value == 2:
                            return True

    return False


# ---------------------------------------------------------------------------
# Sort-call detection
# ---------------------------------------------------------------------------

def _detect_sort_call(node: ast.AST) -> bool:
    """Detect calls to built-in sort / sorted."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == "sorted":
                return True
            if isinstance(child.func, ast.Attribute) and child.func.attr == "sort":
                return True
    return False


# ---------------------------------------------------------------------------
# Main estimation
# ---------------------------------------------------------------------------

def estimate_function_complexity(
    func_node: ast.AST,
    func_name: str,
) -> ComplexityEstimate:
    """Estimate the time complexity of a single function.

    Applies heuristics in priority order:
    1. Divide-and-conquer / merge-sort  → O(n log n)
    2. Exponential recursion (Fibonacci) → O(2^n)
    3. Linear recursion                  → O(n)
    4. Binary search                     → O(log n)
    5. Logarithmic loop                  → O(log n)
    6. Nested loops (depth → O(n^depth)) up to O(n³)
    7. Single loop                       → O(n)
    8. Sort call                         → O(n log n)
    9. No loops / no recursion           → O(1)

    Parameters
    ----------
    func_node : ast.AST
        The ``FunctionDef`` or ``AsyncFunctionDef`` node.
    func_name : str
        Name of the function (used for recursion detection).

    Returns
    -------
    ComplexityEstimate
    """
    start_line = getattr(func_node, "lineno", 0)
    end_line = getattr(func_node, "end_lineno", start_line)

    is_recursive = detect_recursive_calls(func_node, func_name)
    recursive_branches = _count_recursive_branches(func_node, func_name) if is_recursive else 0
    loop_nests = analyze_loop_nesting(func_node)
    max_depth = max((ln.depth for ln in loop_nests), default=0)

    # ----- 1. Divide-and-conquer / merge sort -----
    if is_recursive and detect_divide_and_conquer(func_node):
        rule = match_rule("divide_and_conquer")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' uses a divide-and-conquer pattern: "
                f"it splits the input and makes {recursive_branches} recursive calls."
            ),
            loop_nests=loop_nests,
            is_recursive=True,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 2. Exponential recursion (2+ branches, no splitting) -----
    if is_recursive and recursive_branches >= 2:
        rule = match_rule("exponential_recursion")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' makes {recursive_branches} recursive "
                f"calls without clear input splitting — likely exponential."
            ),
            loop_nests=loop_nests,
            is_recursive=True,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 3. Binary search -----
    if detect_binary_search_pattern(func_node):
        rule = match_rule("binary_search")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' contains a binary search pattern "
                f"(midpoint calculation with bound adjustments)."
            ),
            loop_nests=loop_nests,
            is_recursive=is_recursive,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 4. Linear recursion -----
    if is_recursive and recursive_branches == 1:
        rule = match_rule("linear_recursion")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' makes a single recursive call — "
                f"likely linear recursion."
            ),
            loop_nests=loop_nests,
            is_recursive=True,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 5. Logarithmic loop -----
    if _detect_logarithmic_loop(func_node):
        rule = match_rule("logarithmic_loop")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' contains a loop whose counter "
                f"doubles or halves each iteration."
            ),
            loop_nests=loop_nests,
            is_recursive=False,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 6. Nested loops -----
    if max_depth >= 3:
        rule = match_rule("nested_loop_3")
        confidence = rule.confidence
        # Lower confidence if any loop has a break
        if any(ln.has_break for ln in loop_nests):
            confidence *= 0.7
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=round(confidence, 2),
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' has {max_depth}-deep nested loops."
            ),
            loop_nests=loop_nests,
            is_recursive=False,
            start_line=start_line,
            end_line=end_line,
        )

    if max_depth == 2:
        rule = match_rule("nested_loop_2")
        confidence = rule.confidence
        if any(ln.has_break for ln in loop_nests):
            confidence *= 0.7
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=round(confidence, 2),
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' has doubly-nested loops."
            ),
            loop_nests=loop_nests,
            is_recursive=False,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 7. Sort call -----
    if _detect_sort_call(func_node):
        rule = match_rule("efficient_sort")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' calls a built-in sort function."
            ),
            loop_nests=loop_nests,
            is_recursive=False,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 8. Single loop -----
    if max_depth == 1:
        rule = match_rule("single_loop")
        return ComplexityEstimate(
            function_name=func_name,
            complexity_class=rule.complexity_class,
            confidence=rule.confidence,
            matched_pattern=rule.pattern_name,
            explanation=(
                f"Function '{func_name}' contains a single (non-nested) loop."
            ),
            loop_nests=loop_nests,
            is_recursive=False,
            start_line=start_line,
            end_line=end_line,
        )

    # ----- 9. Constant -----
    rule = match_rule("constant_operation")
    return ComplexityEstimate(
        function_name=func_name,
        complexity_class=rule.complexity_class,
        confidence=rule.confidence,
        matched_pattern=rule.pattern_name,
        explanation=(
            f"Function '{func_name}' has no loops or recursion — "
            f"constant time."
        ),
        loop_nests=[],
        is_recursive=False,
        start_line=start_line,
        end_line=end_line,
    )


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def estimate_all_functions(module_node: ast.Module) -> List[ComplexityEstimate]:
    """Estimate complexity for every function in a module.

    Parameters
    ----------
    module_node : ast.Module
        Parsed module AST.

    Returns
    -------
    List[ComplexityEstimate]
    """
    estimates: List[ComplexityEstimate] = []
    for node in ast.walk(module_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            est = estimate_function_complexity(node, node.name)
            estimates.append(est)
    return estimates
