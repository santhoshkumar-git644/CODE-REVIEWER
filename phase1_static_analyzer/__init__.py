"""
Phase 1: Static Code Analyzer
==============================

A comprehensive static analysis toolkit that parses source code into ASTs,
walks the tree to extract structural information, and computes software
quality metrics such as cyclomatic complexity, nesting depth, and function
length.

Supported languages: Python (via built-in ast module).
Extensible architecture for C, Java, and JavaScript grammars.
"""

from phase1_static_analyzer.parser import (
    init_parser,
    parse_code,
    get_root_node,
    detect_language,
    SUPPORTED_LANGUAGES,
)
from phase1_static_analyzer.ast_walker import (
    walk_tree,
    find_functions,
    find_loops,
    find_conditions,
    find_variables,
    get_nesting_depth,
    FunctionInfo,
    LoopInfo,
    ConditionInfo,
    VariableInfo,
)
from phase1_static_analyzer.metrics import (
    compute_function_count,
    compute_max_nesting_depth,
    compute_total_lines,
    compute_avg_function_length,
    compute_cyclomatic_complexity,
    compute_all_metrics,
    MetricsReport,
)

__version__ = "1.0.0"
__author__ = "AI Code Review Team"

__all__ = [
    # parser
    "init_parser",
    "parse_code",
    "get_root_node",
    "detect_language",
    "SUPPORTED_LANGUAGES",
    # ast_walker
    "walk_tree",
    "find_functions",
    "find_loops",
    "find_conditions",
    "find_variables",
    "get_nesting_depth",
    "FunctionInfo",
    "LoopInfo",
    "ConditionInfo",
    "VariableInfo",
    # metrics
    "compute_function_count",
    "compute_max_nesting_depth",
    "compute_total_lines",
    "compute_avg_function_length",
    "compute_cyclomatic_complexity",
    "compute_all_metrics",
    "MetricsReport",
]
