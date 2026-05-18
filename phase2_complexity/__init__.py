"""
Phase 2: Time Complexity Estimator
====================================

Analyses loop structures, recursion patterns, and algorithmic idioms in
parsed ASTs to produce per-function Big-O complexity estimates with
confidence scores.

Provides rule-based pattern matching for common algorithmic patterns
including single loops, nested loops, binary search, divide-and-conquer,
and recursive algorithms.
"""

from phase2_complexity.complexity_rules import (
    ComplexityRule,
    match_rule,
    get_all_rules,
    RULES,
)
from phase2_complexity.loop_analyzer import (
    LoopNestInfo,
    ComplexityEstimate,
    analyze_loop_nesting,
    detect_recursive_calls,
    detect_binary_search_pattern,
    detect_divide_and_conquer,
    estimate_function_complexity,
)
from phase2_complexity.complexity_report import (
    generate_report,
    generate_json_report,
    format_complexity_badge,
)

__version__ = "1.0.0"
__author__ = "AI Code Review Team"

__all__ = [
    # complexity_rules
    "ComplexityRule",
    "match_rule",
    "get_all_rules",
    "RULES",
    # loop_analyzer
    "LoopNestInfo",
    "ComplexityEstimate",
    "analyze_loop_nesting",
    "detect_recursive_calls",
    "detect_binary_search_pattern",
    "detect_divide_and_conquer",
    "estimate_function_complexity",
    # complexity_report
    "generate_report",
    "generate_json_report",
    "format_complexity_badge",
]
