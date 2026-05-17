"""
parser.py — Source Code Parser
===============================

Provides a unified interface for parsing source code into Abstract Syntax
Trees (ASTs). Uses Python's built-in ``ast`` module for Python source code
and provides a pluggable architecture for adding tree-sitter grammars for
C, Java, and JavaScript.

When tree-sitter is unavailable the module falls back to a lightweight
regex-based tokeniser that extracts structural information (functions,
loops, conditions) from non-Python languages.

Public API
----------
- ``init_parser(language)``   — create a language-specific parser instance
- ``parse_code(code, language)`` — parse a source string into an AST/tree
- ``get_root_node(tree)``     — retrieve the root node of a parsed tree
- ``detect_language(filepath)`` — guess language from file extension

Constants
---------
- ``SUPPORTED_LANGUAGES`` — frozenset of supported language identifiers
"""

from __future__ import annotations

import ast
import re
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: frozenset = frozenset({"python", "c", "java", "javascript"})

_EXTENSION_MAP: Dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".c": "c",
    ".h": "c",
    ".java": "java",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
}


# ---------------------------------------------------------------------------
# Data structures for generic (non-Python) AST representation
# ---------------------------------------------------------------------------

@dataclass
class GenericNode:
    """A lightweight AST node used for non-Python languages when tree-sitter
    is not available.  Mirrors the minimal interface expected by the
    ast_walker module.
    """

    node_type: str
    text: str = ""
    start_line: int = 0
    end_line: int = 0
    children: List["GenericNode"] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Convenience helpers so the walker can treat this like an ast.AST node
    @property
    def lineno(self) -> int:
        return self.start_line

    @property
    def end_lineno(self) -> int:
        return self.end_line

    @property
    def name(self) -> str:
        return self.attributes.get("name", "")


@dataclass
class GenericTree:
    """Wrapper around a root ``GenericNode`` to unify the return type of
    ``parse_code`` across languages.
    """

    root: GenericNode
    source: str
    language: str


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(filepath: str) -> str:
    """Detect the programming language from a file extension.

    Parameters
    ----------
    filepath : str
        Path to the source file (only the extension is inspected).

    Returns
    -------
    str
        Language identifier (e.g. ``"python"``, ``"java"``).

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext not in _EXTENSION_MAP:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported extensions: {sorted(_EXTENSION_MAP.keys())}"
        )
    return _EXTENSION_MAP[ext]


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

class _ParserRegistry:
    """Internal registry that caches parser instances per language."""

    def __init__(self) -> None:
        self._parsers: Dict[str, Any] = {}

    def get_or_create(self, language: str) -> Any:
        if language not in self._parsers:
            self._parsers[language] = _create_parser(language)
        return self._parsers[language]


_registry = _ParserRegistry()


# ---------------------------------------------------------------------------
# Python-specific parser (built-in ast)
# ---------------------------------------------------------------------------

class _PythonParser:
    """Wraps the built-in ``ast`` module."""

    language = "python"

    def parse(self, source: str) -> ast.AST:
        """Parse Python source code into an ``ast.Module`` node.

        Parameters
        ----------
        source : str
            Python source code.

        Returns
        -------
        ast.AST
            The parsed AST.

        Raises
        ------
        SyntaxError
            If the source contains invalid Python syntax.
        """
        try:
            tree = ast.parse(source, mode="exec")
            return tree
        except SyntaxError as exc:
            logger.error("Python syntax error at line %s: %s", exc.lineno, exc.msg)
            raise


# ---------------------------------------------------------------------------
# Generic regex-based parser for C / Java / JavaScript
# ---------------------------------------------------------------------------

_FUNCTION_PATTERNS: Dict[str, re.Pattern] = {
    "c": re.compile(
        r"^\s*(?:(?:static|inline|extern|unsigned|signed|const|volatile)\s+)*"
        r"(?:void|int|float|double|char|long|short|bool|size_t|\w+\s*\*?)\s+"
        r"(\w+)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    ),
    "java": re.compile(
        r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*"
        r"(?:void|int|float|double|char|long|short|boolean|byte|String|\w+(?:<[^>]+>)?)\s+"
        r"(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*\{",
        re.MULTILINE,
    ),
    "javascript": re.compile(
        r"(?:^\s*(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{"
        r"|^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)\s*)"
        ,
        re.MULTILINE,
    ),
}

_LOOP_PATTERN = re.compile(
    r"\b(for|while|do)\s*[\(\{]", re.MULTILINE
)
_CONDITION_PATTERN = re.compile(
    r"\b(if|else\s+if|else|switch)\b", re.MULTILINE
)
_VARIABLE_PATTERN_C = re.compile(
    r"^\s*(?:int|float|double|char|long|short|unsigned|signed|bool|size_t|void\s*\*)\s+(\w+)\s*[;=]",
    re.MULTILINE,
)
_VARIABLE_PATTERN_JAVA = re.compile(
    r"^\s*(?:int|float|double|char|long|short|byte|boolean|String|\w+(?:<[^>]+>)?)\s+(\w+)\s*[;=]",
    re.MULTILINE,
)
_VARIABLE_PATTERN_JS = re.compile(
    r"^\s*(?:const|let|var)\s+(\w+)\s*[;=]",
    re.MULTILINE,
)

_VARIABLE_PATTERNS: Dict[str, re.Pattern] = {
    "c": _VARIABLE_PATTERN_C,
    "java": _VARIABLE_PATTERN_JAVA,
    "javascript": _VARIABLE_PATTERN_JS,
}


class _GenericParser:
    """Regex-based structural parser for C, Java, and JavaScript.

    This is a *best-effort* fallback used when tree-sitter native grammars
    are not compiled and available.  It extracts function boundaries, loops,
    conditionals, and variable declarations using regular expressions and
    simple brace-matching.
    """

    def __init__(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES or language == "python":
            raise ValueError(f"GenericParser does not handle '{language}'")
        self.language = language

    # ----- helpers -----

    @staticmethod
    def _find_matching_brace(source: str, open_pos: int) -> int:
        """Return the index of the closing ``}`` that matches the ``{`` at
        *open_pos*, or ``-1`` if not found.
        """
        depth = 0
        in_string: Optional[str] = None
        i = open_pos
        while i < len(source):
            ch = source[i]
            # Handle string literals (simplified — no escape handling)
            if in_string:
                if ch == in_string and (i == 0 or source[i - 1] != "\\"):
                    in_string = None
            else:
                if ch in ('"', "'", "`"):
                    in_string = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    @staticmethod
    def _line_of(source: str, index: int) -> int:
        """Return 1-based line number for a character index."""
        return source[:index].count("\n") + 1

    # ----- main parse -----

    def parse(self, source: str) -> GenericTree:
        """Parse *source* into a ``GenericTree``."""
        root = GenericNode(node_type="module", text="", start_line=1,
                           end_line=source.count("\n") + 1)

        # --- functions ---
        func_pat = _FUNCTION_PATTERNS.get(self.language)
        if func_pat:
            for match in func_pat.finditer(source):
                # Pick the first non-None group as function name
                func_name = next((g for g in match.groups() if g), "anonymous")
                start_idx = match.start()
                brace_idx = source.find("{", match.end() - 1)
                if brace_idx == -1:
                    brace_idx = source.find("{", start_idx)
                end_idx = self._find_matching_brace(source, brace_idx) if brace_idx != -1 else -1
                start_line = self._line_of(source, start_idx)
                end_line = self._line_of(source, end_idx) if end_idx != -1 else start_line
                body_text = source[brace_idx:end_idx + 1] if brace_idx != -1 and end_idx != -1 else ""
                func_node = GenericNode(
                    node_type="function_def",
                    text=body_text,
                    start_line=start_line,
                    end_line=end_line,
                    attributes={"name": func_name},
                )
                # Find loops inside the function body
                self._extract_loops(body_text, start_line, func_node)
                # Find conditions inside the function body
                self._extract_conditions(body_text, start_line, func_node)
                # Find variables inside the function body
                self._extract_variables(body_text, start_line, func_node)
                root.children.append(func_node)

        # --- top-level loops ---
        self._extract_loops(source, 1, root)
        # --- top-level conditions ---
        self._extract_conditions(source, 1, root)
        # --- top-level variables ---
        self._extract_variables(source, 1, root)

        return GenericTree(root=root, source=source, language=self.language)

    def _extract_loops(self, text: str, base_line: int, parent: GenericNode) -> None:
        for match in _LOOP_PATTERN.finditer(text):
            keyword = match.group(1)
            line = base_line + text[:match.start()].count("\n")
            loop_node = GenericNode(
                node_type="loop",
                text=keyword,
                start_line=line,
                end_line=line,
                attributes={"keyword": keyword},
            )
            parent.children.append(loop_node)

    def _extract_conditions(self, text: str, base_line: int, parent: GenericNode) -> None:
        for match in _CONDITION_PATTERN.finditer(text):
            keyword = match.group(1).strip()
            line = base_line + text[:match.start()].count("\n")
            cond_node = GenericNode(
                node_type="condition",
                text=keyword,
                start_line=line,
                end_line=line,
                attributes={"keyword": keyword},
            )
            parent.children.append(cond_node)

    def _extract_variables(self, text: str, base_line: int, parent: GenericNode) -> None:
        var_pat = _VARIABLE_PATTERNS.get(self.language)
        if not var_pat:
            return
        for match in var_pat.finditer(text):
            var_name = match.group(1)
            line = base_line + text[:match.start()].count("\n")
            var_node = GenericNode(
                node_type="variable",
                text=var_name,
                start_line=line,
                end_line=line,
                attributes={"name": var_name},
            )
            parent.children.append(var_node)


# ---------------------------------------------------------------------------
# Tree-sitter loader (optional, graceful degradation)
# ---------------------------------------------------------------------------

_TREE_SITTER_AVAILABLE = False

try:
    import tree_sitter  # type: ignore[import-untyped]
    from tree_sitter import Language, Parser as TSParser  # type: ignore[import-untyped]
    _TREE_SITTER_AVAILABLE = True
    logger.info("tree-sitter is available — native grammars will be used when compiled.")
except ImportError:
    logger.debug("tree-sitter not installed; falling back to regex-based parser for non-Python languages.")


def _try_load_tree_sitter_parser(language: str) -> Optional[Any]:
    """Attempt to load a compiled tree-sitter grammar.  Returns ``None`` if
    tree-sitter is not installed or the grammar .so/.dll is not found.
    """
    if not _TREE_SITTER_AVAILABLE:
        return None

    grammar_dir = os.path.join(os.path.dirname(__file__), "grammars")
    lib_name = f"tree-sitter-{language}"
    so_path = os.path.join(grammar_dir, f"{lib_name}.so")
    dll_path = os.path.join(grammar_dir, f"{lib_name}.dll")

    lib_path = so_path if os.path.isfile(so_path) else dll_path if os.path.isfile(dll_path) else None
    if lib_path is None:
        logger.debug("Compiled grammar not found for '%s' at %s", language, grammar_dir)
        return None

    try:
        lang = Language(lib_path, language)
        parser = TSParser()
        parser.set_language(lang)
        logger.info("Loaded tree-sitter grammar for '%s' from %s", language, lib_path)
        return parser
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load tree-sitter grammar for '%s': %s", language, exc)
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _create_parser(language: str) -> Any:
    """Create the most capable parser available for *language*."""
    language = language.lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if language == "python":
        return _PythonParser()

    # Try tree-sitter first, then fall back to regex parser
    ts_parser = _try_load_tree_sitter_parser(language)
    if ts_parser is not None:
        return ts_parser

    return _GenericParser(language)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_parser(language: str) -> Any:
    """Initialise and return a parser for the given *language*.

    Parameters
    ----------
    language : str
        One of ``"python"``, ``"c"``, ``"java"``, ``"javascript"``.

    Returns
    -------
    parser
        A parser object whose ``.parse(source)`` method returns a tree.

    Raises
    ------
    ValueError
        If *language* is not supported.

    Examples
    --------
    >>> p = init_parser("python")
    >>> tree = p.parse("x = 1\\n")
    """
    language = language.lower().strip()
    return _registry.get_or_create(language)


def parse_code(code: str, language: str) -> Any:
    """Parse *code* written in *language* and return the AST / tree.

    Parameters
    ----------
    code : str
        Source code string.
    language : str
        Language identifier.

    Returns
    -------
    ast.Module | GenericTree
        Parsed tree.  For Python this is an ``ast.Module``; for other
        languages it is a ``GenericTree`` (or a tree-sitter ``Tree`` when
        native grammars are available).

    Raises
    ------
    SyntaxError
        If *code* cannot be parsed.
    ValueError
        If *language* is not supported.
    """
    language = language.lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    parser = init_parser(language)
    try:
        return parser.parse(code)
    except SyntaxError:
        raise
    except Exception as exc:
        logger.error("Parsing failed for language '%s': %s", language, exc)
        raise SyntaxError(f"Failed to parse {language} code: {exc}") from exc


def get_root_node(tree: Any) -> Any:
    """Return the root node of *tree*.

    Parameters
    ----------
    tree : ast.Module | GenericTree | tree-sitter Tree
        A parsed tree returned by ``parse_code``.

    Returns
    -------
    ast.AST | GenericNode
        The root node.

    Raises
    ------
    TypeError
        If *tree* is not a recognised tree type.
    """
    # Python built-in ast
    if isinstance(tree, ast.AST):
        return tree

    # Our generic tree
    if isinstance(tree, GenericTree):
        return tree.root

    # tree-sitter Tree (if available)
    if hasattr(tree, "root_node"):
        return tree.root_node

    raise TypeError(f"Unknown tree type: {type(tree).__name__}")
