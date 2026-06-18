"""
complexity_report.py — Complexity Analysis Report Formatter
=============================================================

Formats the per-function ``ComplexityEstimate`` results into human-readable
tables, machine-readable JSON, and badge strings suitable for terminal
output.

Public API
----------
- ``generate_report(estimates)``       — bordered ASCII table
- ``generate_json_report(estimates)``  — dict suitable for ``json.dumps``
- ``format_complexity_badge(class_)``  — coloured badge string
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from phase2_complexity.loop_analyzer import ComplexityEstimate, LoopNestInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI colour helpers (graceful no-op when piped to a file)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"

_COLOURS: Dict[str, str] = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "orange": "\033[38;5;208m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "grey": "\033[90m",
}


def _c(text: str, colour: str, bold: bool = False) -> str:
    """Wrap *text* in ANSI colour codes."""
    prefix = _COLOURS.get(colour, "")
    if bold:
        prefix = _BOLD + prefix
    return f"{prefix}{text}{_RESET}" if prefix else text


# ---------------------------------------------------------------------------
# Complexity → colour mapping
# ---------------------------------------------------------------------------

_COMPLEXITY_COLOUR: Dict[str, str] = {
    "O(1)": "green",
    "O(log n)": "green",
    "O(n)": "cyan",
    "O(n log n)": "yellow",
    "O(n²)": "orange",
    "O(n³)": "red",
    "O(2^n)": "red",
    "O(n!)": "magenta",
}

_COMPLEXITY_SEVERITY: Dict[str, str] = {
    "O(1)": "EXCELLENT",
    "O(log n)": "EXCELLENT",
    "O(n)": "GOOD",
    "O(n log n)": "ACCEPTABLE",
    "O(n²)": "CAUTION",
    "O(n³)": "WARNING",
    "O(2^n)": "DANGER",
    "O(n!)": "CRITICAL",
}

_COMPLEXITY_ORDER: Dict[str, int] = {
    "O(1)": 0,
    "O(log n)": 1,
    "O(n)": 2,
    "O(n log n)": 3,
    "O(n²)": 4,
    "O(n³)": 5,
    "O(2^n)": 6,
    "O(n!)": 7,
}


# ---------------------------------------------------------------------------
# Badge formatter
# ---------------------------------------------------------------------------

def format_complexity_badge(complexity_class: str) -> str:
    """Return a coloured badge string for the given Big-O class.

    Parameters
    ----------
    complexity_class : str
        E.g. ``"O(n)"``, ``"O(n²)"``.

    Returns
    -------
    str
        A terminal-friendly string like ``[O(n²) ⚠ CAUTION]``.

    Examples
    --------
    >>> format_complexity_badge("O(n)")
    '[O(n) ✅ GOOD]'
    """
    colour = _COMPLEXITY_COLOUR.get(complexity_class, "white")
    severity = _COMPLEXITY_SEVERITY.get(complexity_class, "UNKNOWN")

    icons = {
        "EXCELLENT": "✅",
        "GOOD": "✅",
        "ACCEPTABLE": "🟡",
        "CAUTION": "⚠️",
        "WARNING": "🔶",
        "DANGER": "🔴",
        "CRITICAL": "🚨",
    }
    icon = icons.get(severity, "❓")

    badge_text = f"[{complexity_class} {icon} {severity}]"
    return _c(badge_text, colour, bold=True)


# ---------------------------------------------------------------------------
# Confidence bar
# ---------------------------------------------------------------------------

def _confidence_bar(confidence: float, width: int = 10) -> str:
    """Render a small ASCII confidence bar.

    >>> _confidence_bar(0.85)
    '████████░░ 85%'
    """
    filled = round(confidence * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    pct = f"{confidence:.0%}"
    return f"{bar} {pct}"


# ---------------------------------------------------------------------------
# ASCII table report
# ---------------------------------------------------------------------------

def generate_report(estimates: List[ComplexityEstimate]) -> str:
    """Generate a bordered ASCII table summarising complexity estimates.

    Parameters
    ----------
    estimates : List[ComplexityEstimate]
        One estimate per function.

    Returns
    -------
    str
        Multi-line string ready to be printed to the terminal.
    """
    if not estimates:
        return "(no functions to analyse)"

    # Sort by severity (worst first)
    sorted_est = sorted(
        estimates,
        key=lambda e: _COMPLEXITY_ORDER.get(e.complexity_class, 99),
        reverse=True,
    )

    # Column widths
    name_w = max(len(e.function_name) for e in sorted_est)
    name_w = max(name_w, len("Function")) + 2
    cc_w = 12  # "O(n log n)" is 10 chars
    pattern_w = max(len(e.matched_pattern) for e in sorted_est)
    pattern_w = max(pattern_w, len("Pattern")) + 2
    conf_w = 18  # bar + pct
    lines_w = 12

    total_w = name_w + cc_w + pattern_w + conf_w + lines_w + 6  # separators

    border = "+" + "-" * (total_w) + "+"
    double_border = "+" + "=" * (total_w) + "+"

    lines: List[str] = []

    # Title
    lines.append(double_border)
    title = " TIME COMPLEXITY ANALYSIS "
    lines.append("|" + title.center(total_w) + "|")
    lines.append(double_border)

    # Header
    header = (
        "| "
        + "Function".ljust(name_w)
        + "| " + "Complexity".ljust(cc_w)
        + "| " + "Pattern".ljust(pattern_w)
        + "| " + "Confidence".ljust(conf_w)
        + "| " + "Lines".ljust(lines_w)
        + "|"
    )
    lines.append(header)
    lines.append(border)

    # Rows
    for est in sorted_est:
        badge = est.complexity_class
        conf = _confidence_bar(est.confidence)
        line_range = f"L{est.start_line}-{est.end_line}" if est.start_line else "-"
        recursive_flag = " (R)" if est.is_recursive else ""

        row = (
            "| "
            + (est.function_name + recursive_flag).ljust(name_w)
            + "| " + badge.ljust(cc_w)
            + "| " + est.matched_pattern.ljust(pattern_w)
            + "| " + conf.ljust(conf_w)
            + "| " + line_range.ljust(lines_w)
            + "|"
        )
        lines.append(row)

    lines.append(border)

    # Summary statistics
    lines.append("")
    worst = sorted_est[0]
    lines.append(f"  Worst-case complexity: {format_complexity_badge(worst.complexity_class)}")
    lines.append(f"  Functions analysed:    {len(estimates)}")

    recursive_count = sum(1 for e in estimates if e.is_recursive)
    if recursive_count:
        lines.append(f"  Recursive functions:   {recursive_count}")

    # Detailed explanations
    lines.append("")
    lines.append("  Detailed Explanations:")
    lines.append("  " + "-" * 60)
    for est in sorted_est:
        lines.append(f"  {est.function_name}:")
        lines.append(f"    {est.explanation}")
        if est.loop_nests:
            for ln in est.loop_nests:
                loop_desc = " → ".join(ln.loop_types)
                extras = []
                if ln.has_break:
                    extras.append("has break")
                if ln.has_early_return:
                    extras.append("has early return")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                lines.append(f"    Loop chain: {loop_desc} (depth {ln.depth}){extra_str}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def generate_json_report(estimates: List[ComplexityEstimate]) -> Dict[str, Any]:
    """Generate a JSON-serialisable report.

    Parameters
    ----------
    estimates : List[ComplexityEstimate]

    Returns
    -------
    dict
        Dictionary with ``"summary"`` and ``"functions"`` keys.
    """
    if not estimates:
        return {"summary": {"function_count": 0}, "functions": []}

    # Determine worst-case
    worst = max(
        estimates,
        key=lambda e: _COMPLEXITY_ORDER.get(e.complexity_class, 99),
    )

    summary: Dict[str, Any] = {
        "function_count": len(estimates),
        "worst_case_complexity": worst.complexity_class,
        "worst_case_function": worst.function_name,
        "recursive_functions": sum(1 for e in estimates if e.is_recursive),
        "average_confidence": round(
            sum(e.confidence for e in estimates) / len(estimates), 2
        ),
    }

    # Complexity distribution
    distribution: Dict[str, int] = {}
    for est in estimates:
        distribution[est.complexity_class] = distribution.get(est.complexity_class, 0) + 1
    summary["complexity_distribution"] = distribution

    # Per-function details
    functions: List[Dict[str, Any]] = []
    for est in estimates:
        func_dict: Dict[str, Any] = {
            "name": est.function_name,
            "complexity_class": est.complexity_class,
            "confidence": est.confidence,
            "matched_pattern": est.matched_pattern,
            "explanation": est.explanation,
            "is_recursive": est.is_recursive,
            "start_line": est.start_line,
            "end_line": est.end_line,
            "loop_nests": [],
        }
        for ln in est.loop_nests:
            func_dict["loop_nests"].append({
                "depth": ln.depth,
                "loop_types": ln.loop_types,
                "start_line": ln.start_line,
                "end_line": ln.end_line,
                "has_break": ln.has_break,
                "has_early_return": ln.has_early_return,
            })
        functions.append(func_dict)

    return {
        "summary": summary,
        "functions": functions,
    }


def generate_json_string(estimates: List[ComplexityEstimate], indent: int = 2) -> str:
    """Convenience wrapper that returns a JSON string.

    Parameters
    ----------
    estimates : List[ComplexityEstimate]
    indent : int

    Returns
    -------
    str
    """
    data = generate_json_report(estimates)
    return json.dumps(data, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def generate_csv_report(estimates: List[ComplexityEstimate]) -> str:
    """Generate a CSV-formatted report.

    Parameters
    ----------
    estimates : List[ComplexityEstimate]

    Returns
    -------
    str
        CSV text with header row.
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "function_name",
        "complexity_class",
        "confidence",
        "matched_pattern",
        "is_recursive",
        "start_line",
        "end_line",
        "max_loop_depth",
        "explanation",
    ])

    for est in estimates:
        max_loop = max((ln.depth for ln in est.loop_nests), default=0)
        writer.writerow([
            est.function_name,
            est.complexity_class,
            f"{est.confidence:.2f}",
            est.matched_pattern,
            est.is_recursive,
            est.start_line,
            est.end_line,
            max_loop,
            est.explanation,
        ])

    return output.getvalue()


# ---------------------------------------------------------------------------
# Markdown report (for integration with documentation pipelines)
# ---------------------------------------------------------------------------

def generate_markdown_report(estimates: List[ComplexityEstimate]) -> str:
    """Generate a Markdown-formatted complexity report.

    Parameters
    ----------
    estimates : List[ComplexityEstimate]

    Returns
    -------
    str
        Markdown text.
    """
    if not estimates:
        return "## Time Complexity Analysis\n\n_No functions found._\n"

    sorted_est = sorted(
        estimates,
        key=lambda e: _COMPLEXITY_ORDER.get(e.complexity_class, 99),
        reverse=True,
    )

    worst = sorted_est[0]

    lines: List[str] = []
    lines.append("## Time Complexity Analysis")
    lines.append("")
    lines.append(f"**Worst-case complexity:** `{worst.complexity_class}` "
                 f"({_COMPLEXITY_SEVERITY.get(worst.complexity_class, 'UNKNOWN')})")
    lines.append(f"**Functions analysed:** {len(estimates)}")
    lines.append("")

    # Table
    lines.append("| Function | Complexity | Pattern | Confidence | Recursive |")
    lines.append("|----------|-----------|---------|------------|-----------|")
    for est in sorted_est:
        rec = "Yes" if est.is_recursive else "No"
        lines.append(
            f"| `{est.function_name}` | `{est.complexity_class}` | "
            f"{est.matched_pattern} | {est.confidence:.0%} | {rec} |"
        )

    lines.append("")

    # Details
    lines.append("### Detailed Explanations")
    lines.append("")
    for est in sorted_est:
        lines.append(f"#### `{est.function_name}`")
        lines.append(f"- **Complexity:** `{est.complexity_class}`")
        lines.append(f"- **Confidence:** {est.confidence:.0%}")
        lines.append(f"- **Pattern:** {est.matched_pattern}")
        lines.append(f"- {est.explanation}")
        if est.loop_nests:
            lines.append("- **Loop chains:**")
            for ln in est.loop_nests:
                chain = " → ".join(ln.loop_types)
                lines.append(f"  - `{chain}` (depth {ln.depth})")
        lines.append("")

    return "\n".join(lines)
