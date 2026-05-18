"""
metrics.py — Code Quality Metrics
===================================

Computes a comprehensive set of static analysis metrics for a given source
file, including:

- Total line count (total, code, blank, comment)
- Function count and average function length
- Maximum nesting depth
- Cyclomatic complexity (McCabe)
- Maintainability index (simplified)

All metrics are bundled into a ``MetricsReport`` dataclass that can be
serialised to JSON or formatted as a table.

Public API
----------
- ``compute_function_count(functions)``
- ``compute_max_nesting_depth(node)``
- ``compute_total_lines(code)``
- ``compute_avg_function_length(functions)``
- ``compute_cyclomatic_complexity(node)``
- ``compute_all_metrics(code, language)``
- ``MetricsReport``
"""

from __future__ import annotations

import ast
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from phase1_static_analyzer.parser import GenericNode, parse_code, get_root_node
from phase1_static_analyzer.ast_walker import (
    FunctionInfo,
    find_functions,
    find_loops,
    find_conditions,
    find_variables,
    get_nesting_depth,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricsReport
# ---------------------------------------------------------------------------

@dataclass
class MetricsReport:
    """Aggregated code-quality metrics for a single source file."""

    # Line counts
    total_lines: int = 0
    code_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0

    # Structural counts
    function_count: int = 0
    class_count: int = 0
    loop_count: int = 0
    condition_count: int = 0
    variable_count: int = 0

    # Complexity
    max_nesting_depth: int = 0
    cyclomatic_complexity: int = 1
    avg_function_length: float = 0.0
    max_function_length: int = 0

    # Maintainability (simplified MI — higher is better, max ~171)
    maintainability_index: float = 0.0

    # Per-function breakdown
    per_function_complexity: Dict[str, int] = field(default_factory=dict)
    per_function_length: Dict[str, int] = field(default_factory=dict)

    # Language
    language: str = "unknown"

    # Risk assessment
    risk_level: str = "low"  # "low" | "medium" | "high"
    risk_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_table(self) -> str:
        """Render a human-readable table."""
        return _format_metrics_table(self)


# ---------------------------------------------------------------------------
# Line counting helpers
# ---------------------------------------------------------------------------

_COMMENT_PREFIXES: Dict[str, List[str]] = {
    "python": ["#"],
    "c": ["//", "/*"],
    "java": ["//", "/*"],
    "javascript": ["//", "/*"],
}


def _count_lines(code: str, language: str = "python") -> Dict[str, int]:
    """Count total, code, blank, and comment lines."""
    lines = code.splitlines()
    total = len(lines)
    blank = 0
    comment = 0
    in_block_comment = False
    prefixes = _COMMENT_PREFIXES.get(language, ["#"])

    for raw_line in lines:
        stripped = raw_line.strip()

        # Handle block comments for C-family languages
        if language in ("c", "java", "javascript"):
            if in_block_comment:
                comment += 1
                if "*/" in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith("/*"):
                comment += 1
                if "*/" not in stripped:
                    in_block_comment = True
                continue

        if not stripped:
            blank += 1
        elif any(stripped.startswith(p) for p in prefixes):
            comment += 1

    code_lines = total - blank - comment
    return {
        "total": total,
        "code": max(code_lines, 0),
        "blank": blank,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_function_count(functions: List[FunctionInfo]) -> int:
    """Return the number of functions.

    Parameters
    ----------
    functions : List[FunctionInfo]
        Functions extracted by ``find_functions``.

    Returns
    -------
    int
    """
    return len(functions)


def compute_max_nesting_depth(node: Any) -> int:
    """Return the maximum control-flow nesting depth.

    Delegates to ``ast_walker.get_nesting_depth``.

    Parameters
    ----------
    node : ast.AST | GenericNode

    Returns
    -------
    int
    """
    return get_nesting_depth(node)


def compute_total_lines(code: str) -> int:
    """Return the total number of lines in *code*.

    Parameters
    ----------
    code : str

    Returns
    -------
    int
    """
    if not code:
        return 0
    return len(code.splitlines())


def compute_avg_function_length(functions: List[FunctionInfo]) -> float:
    """Return the average function length in lines.

    Parameters
    ----------
    functions : List[FunctionInfo]

    Returns
    -------
    float
        Average length, or ``0.0`` if no functions exist.
    """
    if not functions:
        return 0.0
    total_length = sum(f.length for f in functions)
    return round(total_length / len(functions), 2)


# ---------------------------------------------------------------------------
# Cyclomatic complexity
# ---------------------------------------------------------------------------

def _python_cyclomatic_complexity(node: ast.AST) -> int:
    """Compute McCabe cyclomatic complexity for a Python AST.

    CC = 1 + number of decision points.

    Decision points: ``if``, ``elif``, ``for``, ``while``, ``except``,
    ``with``, ``assert``, ``and``, ``or``, comprehension ``if`` clauses.
    """
    complexity = 1  # base complexity

    for child in ast.walk(node):
        if isinstance(child, ast.If):
            complexity += 1
        elif isinstance(child, ast.For):
            complexity += 1
        elif isinstance(child, ast.While):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.With):
            complexity += 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Each ``and``/``or`` adds one decision path
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            # List/set/dict comprehension conditions
            complexity += len(child.ifs)

    return complexity


def _per_function_cyclomatic(root: ast.AST) -> Dict[str, int]:
    """Compute cyclomatic complexity per function in a Python AST."""
    result: Dict[str, int] = {}
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = _python_cyclomatic_complexity(node)
    return result


def _generic_cyclomatic_complexity(node: GenericNode) -> int:
    """Approximate cyclomatic complexity for a GenericNode tree.

    Counts loops and conditions as decision points.
    """
    complexity = 1
    from phase1_static_analyzer.ast_walker import _generic_walk  # avoid top-level circular

    for child in _generic_walk(node):
        if child.node_type == "loop":
            complexity += 1
        elif child.node_type == "condition":
            complexity += 1
    return complexity


def compute_cyclomatic_complexity(node: Any) -> int:
    """Compute cyclomatic complexity for the given AST root.

    Parameters
    ----------
    node : ast.AST | GenericNode

    Returns
    -------
    int
        McCabe cyclomatic complexity (minimum 1).
    """
    if isinstance(node, ast.AST):
        return _python_cyclomatic_complexity(node)
    if isinstance(node, GenericNode):
        return _generic_cyclomatic_complexity(node)
    logger.warning("Unsupported node type for cyclomatic complexity: %s", type(node).__name__)
    return 1


# ---------------------------------------------------------------------------
# Maintainability Index (simplified)
# ---------------------------------------------------------------------------

def _compute_maintainability_index(
    halstead_volume: float,
    cyclomatic: int,
    lines_of_code: int,
) -> float:
    """Simplified Maintainability Index.

    MI = 171 − 5.2 × ln(V) − 0.23 × CC − 16.2 × ln(LOC)

    Clamped to [0, 100] after scaling by 171.
    """
    if halstead_volume <= 0:
        halstead_volume = 1.0
    if lines_of_code <= 0:
        lines_of_code = 1

    mi = (
        171.0
        - 5.2 * math.log(halstead_volume)
        - 0.23 * cyclomatic
        - 16.2 * math.log(lines_of_code)
    )
    # Scale to 0-100
    mi_scaled = max(0.0, mi) * 100.0 / 171.0
    return round(min(mi_scaled, 100.0), 2)


def _estimate_halstead_volume(code: str) -> float:
    """Very rough Halstead volume estimate based on token counts.

    This is a simplified heuristic — a proper implementation would use
    a full tokeniser to separate operators and operands.
    """
    # Count unique "words" as a proxy for distinct operands/operators
    import re
    tokens = re.findall(r"[A-Za-z_]\w*|[^\s\w]", code)
    if not tokens:
        return 1.0
    n = len(tokens)                # total tokens
    n_unique = len(set(tokens))    # unique tokens
    if n_unique <= 1:
        return 1.0
    volume = n * math.log2(n_unique)
    return max(volume, 1.0)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

def _assess_risk(report: MetricsReport) -> None:
    """Fill in ``risk_level`` and ``risk_reasons`` on *report*."""
    reasons: List[str] = []

    if report.cyclomatic_complexity > 20:
        reasons.append(f"Very high cyclomatic complexity ({report.cyclomatic_complexity})")
    elif report.cyclomatic_complexity > 10:
        reasons.append(f"High cyclomatic complexity ({report.cyclomatic_complexity})")

    if report.max_nesting_depth > 5:
        reasons.append(f"Excessive nesting depth ({report.max_nesting_depth})")
    elif report.max_nesting_depth > 3:
        reasons.append(f"Deep nesting ({report.max_nesting_depth})")

    if report.max_function_length > 100:
        reasons.append(f"Very long function ({report.max_function_length} lines)")
    elif report.max_function_length > 50:
        reasons.append(f"Long function ({report.max_function_length} lines)")

    if report.maintainability_index < 20:
        reasons.append(f"Low maintainability index ({report.maintainability_index})")
    elif report.maintainability_index < 40:
        reasons.append(f"Below-average maintainability ({report.maintainability_index})")

    if report.function_count == 0 and report.code_lines > 50:
        reasons.append("No functions in a file with 50+ lines — consider refactoring")

    # Determine level
    high_indicators = sum(1 for r in reasons if "Very" in r or "Excessive" in r or "Low maint" in r)
    if high_indicators >= 1 or len(reasons) >= 3:
        report.risk_level = "high"
    elif len(reasons) >= 1:
        report.risk_level = "medium"
    else:
        report.risk_level = "low"

    report.risk_reasons = reasons


# ---------------------------------------------------------------------------
# Class counting (Python only)
# ---------------------------------------------------------------------------

def _count_classes(node: Any) -> int:
    """Count class definitions."""
    if isinstance(node, ast.AST):
        return sum(1 for n in ast.walk(node) if isinstance(n, ast.ClassDef))
    return 0


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_all_metrics(code: str, language: str = "python") -> MetricsReport:
    """Parse *code* and compute every available metric.

    Parameters
    ----------
    code : str
        Source code string.
    language : str
        Language identifier (default ``"python"``).

    Returns
    -------
    MetricsReport
        Fully populated metrics report.

    Raises
    ------
    SyntaxError
        If the code cannot be parsed.
    ValueError
        If the language is unsupported.
    """
    language = language.lower().strip()

    # Parse
    tree = parse_code(code, language)
    root = get_root_node(tree)

    # Line counts
    line_info = _count_lines(code, language)

    # Structural extraction
    functions = find_functions(root)
    loops = find_loops(root)
    conditions = find_conditions(root)
    variables = find_variables(root)

    # Per-function metrics
    func_lengths: Dict[str, int] = {f.name: f.length for f in functions}
    max_func_length = max(func_lengths.values()) if func_lengths else 0

    # Cyclomatic complexity
    total_cc = compute_cyclomatic_complexity(root)
    per_func_cc: Dict[str, int] = {}
    if isinstance(root, ast.AST):
        per_func_cc = _per_function_cyclomatic(root)
    else:
        # For generic nodes approximate per-function CC
        for fn_node in _iter_function_nodes(root):
            fn_name = fn_node.attributes.get("name", "unknown")
            per_func_cc[fn_name] = _generic_cyclomatic_complexity(fn_node)

    # Nesting depth
    max_depth = compute_max_nesting_depth(root)

    # Maintainability index
    halstead_vol = _estimate_halstead_volume(code)
    mi = _compute_maintainability_index(halstead_vol, total_cc, line_info["code"])

    report = MetricsReport(
        total_lines=line_info["total"],
        code_lines=line_info["code"],
        blank_lines=line_info["blank"],
        comment_lines=line_info["comment"],
        function_count=len(functions),
        class_count=_count_classes(root),
        loop_count=len(loops),
        condition_count=len(conditions),
        variable_count=len(variables),
        max_nesting_depth=max_depth,
        cyclomatic_complexity=total_cc,
        avg_function_length=compute_avg_function_length(functions),
        max_function_length=max_func_length,
        maintainability_index=mi,
        per_function_complexity=per_func_cc,
        per_function_length=func_lengths,
        language=language,
    )

    _assess_risk(report)

    return report


def _iter_function_nodes(root: GenericNode):
    """Yield all function-def GenericNodes under *root*."""
    from phase1_static_analyzer.ast_walker import _generic_walk
    for node in _generic_walk(root):
        if node.node_type == "function_def":
            yield node


# ---------------------------------------------------------------------------
# Table formatting (no external dependencies)
# ---------------------------------------------------------------------------

def _format_metrics_table(report: MetricsReport) -> str:
    """Render a ``MetricsReport`` as a human-readable bordered table."""

    # Risk badge
    risk_badge = {
        "low": "[LOW RISK]",
        "medium": "[MEDIUM RISK]",
        "high": "[HIGH RISK]",
    }.get(report.risk_level, "[UNKNOWN]")

    rows = [
        ("Language", report.language.capitalize()),
        ("Total Lines", str(report.total_lines)),
        ("Code Lines", str(report.code_lines)),
        ("Blank Lines", str(report.blank_lines)),
        ("Comment Lines", str(report.comment_lines)),
        ("", ""),
        ("Functions", str(report.function_count)),
        ("Classes", str(report.class_count)),
        ("Loops", str(report.loop_count)),
        ("Conditions", str(report.condition_count)),
        ("Variables", str(report.variable_count)),
        ("", ""),
        ("Cyclomatic Complexity", str(report.cyclomatic_complexity)),
        ("Max Nesting Depth", str(report.max_nesting_depth)),
        ("Avg Function Length", f"{report.avg_function_length} lines"),
        ("Max Function Length", f"{report.max_function_length} lines"),
        ("Maintainability Index", f"{report.maintainability_index} / 100"),
        ("", ""),
        ("Risk Level", risk_badge),
    ]

    # Determine column widths
    label_width = max(len(r[0]) for r in rows) + 2
    value_width = max(len(r[1]) for r in rows) + 2
    total_width = label_width + value_width + 3  # 3 for "| " and " |" and "|"

    border = "+" + "-" * (label_width) + "+" + "-" * (value_width) + "+"
    title_border = "+" + "=" * (total_width - 2) + "+"

    lines: List[str] = []
    lines.append(title_border)
    title = " STATIC ANALYSIS REPORT "
    lines.append("|" + title.center(total_width - 2) + "|")
    lines.append(title_border)

    for label, value in rows:
        if label == "" and value == "":
            lines.append(border)
        else:
            lines.append(
                "|" + f" {label}".ljust(label_width) + "|" + f" {value}".ljust(value_width) + "|"
            )

    lines.append(border)

    # Per-function breakdown
    if report.per_function_complexity or report.per_function_length:
        lines.append("")
        lines.append("Per-Function Breakdown:")
        func_header = f"  {'Function':<30} {'CC':>4}  {'Length':>7}"
        lines.append(func_header)
        lines.append("  " + "-" * len(func_header.strip()))
        all_funcs = set(list(report.per_function_complexity.keys()) + list(report.per_function_length.keys()))
        for fname in sorted(all_funcs):
            cc = report.per_function_complexity.get(fname, "-")
            length = report.per_function_length.get(fname, "-")
            length_str = f"{length} lines" if isinstance(length, int) else str(length)
            lines.append(f"  {fname:<30} {str(cc):>4}  {length_str:>7}")

    # Risk reasons
    if report.risk_reasons:
        lines.append("")
        lines.append("Risk Factors:")
        for reason in report.risk_reasons:
            lines.append(f"  ⚠ {reason}")

    lines.append("")
    return "\n".join(lines)
