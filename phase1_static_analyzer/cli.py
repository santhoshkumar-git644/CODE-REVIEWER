"""
cli.py — Command-Line Interface for Static Code Analyzer
==========================================================

Provides an ``argparse``-based CLI that reads a source file, runs the
static analysis pipeline, and outputs the results in one of three formats:

- **table** (default) — a bordered ASCII table printed to stdout
- **json**  — machine-readable JSON
- **csv**   — comma-separated values suitable for spreadsheet import

Usage
-----
::

    python -m phase1_static_analyzer.cli path/to/file.py
    python -m phase1_static_analyzer.cli path/to/file.py --language python
    python -m phase1_static_analyzer.cli path/to/file.py --output json
    python -m phase1_static_analyzer.cli path/to/file.py --output csv -o report.csv

Exit Codes
----------
- ``0`` — analysis completed successfully
- ``1`` — error (file not found, parse error, unsupported language, etc.)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from typing import List, Optional, TextIO

from phase1_static_analyzer.parser import detect_language, SUPPORTED_LANGUAGES
from phase1_static_analyzer.metrics import compute_all_metrics, MetricsReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _output_table(report: MetricsReport, stream: TextIO) -> None:
    """Write the report as a bordered ASCII table."""
    stream.write(report.to_table())
    stream.write("\n")


def _output_json(report: MetricsReport, stream: TextIO) -> None:
    """Write the report as indented JSON."""
    stream.write(report.to_json(indent=2))
    stream.write("\n")


def _output_csv(report: MetricsReport, stream: TextIO) -> None:
    """Write the report as CSV rows (metric, value)."""
    data = report.to_dict()
    writer = csv.writer(stream)
    writer.writerow(["metric", "value"])

    # Flatten nested dicts for CSV
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                writer.writerow([f"{key}.{sub_key}", sub_value])
        elif isinstance(value, list):
            writer.writerow([key, "; ".join(str(v) for v in value)])
        else:
            writer.writerow([key, value])


_FORMATTERS = {
    "table": _output_table,
    "json": _output_json,
    "csv": _output_csv,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="static-analyzer",
        description=(
            "AI Code Review — Phase 1: Static Code Analyzer\n"
            "Parses source files and computes software quality metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s src/main.py\n"
            "  %(prog)s src/main.py --language python --output json\n"
            "  %(prog)s src/Main.java -o report.csv --output csv\n"
        ),
    )

    parser.add_argument(
        "file",
        help="Path to the source file to analyse.",
    )

    parser.add_argument(
        "--language", "-l",
        choices=sorted(SUPPORTED_LANGUAGES),
        default=None,
        help=(
            "Programming language of the source file. "
            "If omitted, the language is auto-detected from the file extension."
        ),
    )

    parser.add_argument(
        "--output", "-f",
        choices=sorted(_FORMATTERS.keys()),
        default="table",
        dest="format",
        help="Output format (default: table).",
    )

    parser.add_argument(
        "-o", "--outfile",
        default=None,
        help="Write output to this file instead of stdout.",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
╔══════════════════════════════════════════════╗
║   AI Code Review — Static Analyzer v1.0.0   ║
╚══════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Run the static analyser CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Resolve file path
    filepath = os.path.abspath(args.file)
    if not os.path.isfile(filepath):
        print(f"Error: file not found — {filepath}", file=sys.stderr)
        return 1

    # Detect language
    language: str = args.language or ""
    if not language:
        try:
            language = detect_language(filepath)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("Hint: use --language to specify the language explicitly.", file=sys.stderr)
            return 1

    # Read source
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            code = fh.read()
    except OSError as exc:
        print(f"Error: cannot read file — {exc}", file=sys.stderr)
        return 1

    if not code.strip():
        print("Warning: file is empty.", file=sys.stderr)

    # Analyse
    try:
        report = compute_all_metrics(code, language)
    except SyntaxError as exc:
        print(f"Syntax error in {filepath}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Output
    stream: TextIO = sys.stdout
    close_stream = False
    if args.outfile:
        try:
            stream = open(args.outfile, "w", encoding="utf-8", newline="")
            close_stream = True
        except OSError as exc:
            print(f"Error: cannot open output file — {exc}", file=sys.stderr)
            return 1

    try:
        # Print banner only for table format on stdout
        if args.format == "table" and not args.outfile:
            print(_BANNER)
            print(f"  Analysing: {filepath}")
            print(f"  Language:  {language.capitalize()}")
            print()

        formatter = _FORMATTERS[args.format]
        formatter(report, stream)

        # Summary line for table format
        if args.format == "table" and not args.outfile:
            _print_summary(report)

    finally:
        if close_stream:
            stream.close()

    return 0


def _print_summary(report: MetricsReport) -> None:
    """Print a one-line summary with risk level."""
    risk_icons = {
        "low": "✅",
        "medium": "⚠️",
        "high": "🔴",
    }
    icon = risk_icons.get(report.risk_level, "❓")

    print(f"\n{icon} Overall Risk: {report.risk_level.upper()}")

    if report.risk_reasons:
        for reason in report.risk_reasons:
            print(f"   → {reason}")
    else:
        print("   No significant risk factors detected.")

    print()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
