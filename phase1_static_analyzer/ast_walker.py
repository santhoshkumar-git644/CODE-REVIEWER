"""
ast_walker.py — AST Node Traversal & Structural Extraction
============================================================

Walks parsed Abstract Syntax Trees and extracts structural information
such as function definitions, loop constructs, conditional branches, and
variable declarations.

Works with both Python ``ast.AST`` nodes and the ``GenericNode`` trees
produced by the regex-based parser for C / Java / JavaScript.

Data Classes
------------
- ``FunctionInfo``  — function name, line range, parameter count, body text
- ``LoopInfo``      — loop keyword, line range, nesting depth
- ``ConditionInfo`` — condition keyword, line range, has-else flag
- ``VariableInfo``  — variable name, line number, scope

Public API
----------
- ``walk_tree(node)``       — recursively yield every node
- ``find_functions(node)``  — collect ``FunctionInfo`` objects
- ``find_loops(node)``      — collect ``LoopInfo`` objects
- ``find_conditions(node)`` — collect ``ConditionInfo`` objects
- ``find_variables(node)``  — collect ``VariableInfo`` objects
- ``get_nesting_depth(node)`` — compute maximum nesting depth
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, Union

# Import GenericNode for isinstance checks — deferred to avoid circular imports
from phase1_static_analyzer.parser import GenericNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    """Metadata about a single function definition."""

    name: str
    start_line: int
    end_line: int
    parameter_count: int = 0
    body_line_count: int = 0
    decorators: List[str] = field(default_factory=list)
    is_method: bool = False
    docstring: Optional[str] = None
    body_text: str = ""

    @property
    def length(self) -> int:
        """Number of lines spanned by the function (inclusive)."""
        return max(self.end_line - self.start_line + 1, 1)


@dataclass
class LoopInfo:
    """Metadata about a loop construct."""

    keyword: str  # "for" | "while" | "do"
    start_line: int
    end_line: int
    nesting_depth: int = 1
    parent_function: Optional[str] = None
    iteration_target: str = ""


@dataclass
class ConditionInfo:
    """Metadata about a conditional branch."""

    keyword: str  # "if" | "elif" | "else if" | "else" | "switch"
    start_line: int
    end_line: int
    has_else: bool = False
    branch_count: int = 1
    parent_function: Optional[str] = None


@dataclass
class VariableInfo:
    """Metadata about a variable declaration / assignment."""

    name: str
    line: int
    scope: str = "local"  # "local" | "global" | "class"
    type_annotation: Optional[str] = None
    parent_function: Optional[str] = None


# ---------------------------------------------------------------------------
# Generic tree walker
# ---------------------------------------------------------------------------

def walk_tree(node: Any) -> Generator[Any, None, None]:
    """Recursively yield *node* and all of its descendants.

    Supports both ``ast.AST`` nodes and ``GenericNode`` instances.

    Parameters
    ----------
    node : ast.AST | GenericNode
        The root node to start walking from.

    Yields
    ------
    ast.AST | GenericNode
        Each node in depth-first order.
    """
    if node is None:
        return

    yield node

    if isinstance(node, ast.AST):
        for child in ast.iter_child_nodes(node):
            yield from walk_tree(child)
    elif isinstance(node, GenericNode):
        for child in node.children:
            yield from walk_tree(child)
    elif hasattr(node, "children"):
        # tree-sitter compatibility
        for child in node.children:
            yield from walk_tree(child)


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------

def _extract_python_functions(root: ast.AST) -> List[FunctionInfo]:
    """Extract function definitions from a Python AST."""
    functions: List[FunctionInfo] = []

    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Parameter count
            args = node.args
            param_count = (
                len(args.args)
                + len(args.posonlyargs)
                + len(args.kwonlyargs)
                + (1 if args.vararg else 0)
                + (1 if args.kwarg else 0)
            )

            # Decorator names
            decorators: List[str] = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    decorators.append(ast.dump(dec))
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Name):
                        decorators.append(dec.func.id)
                    else:
                        decorators.append(ast.dump(dec.func))

            # Docstring
            docstring: Optional[str] = None
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant,))
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value

            # Body line count
            start = node.lineno
            end = node.end_lineno if node.end_lineno else start
            body_lines = end - start

            # Determine if this is a method (nested inside a ClassDef)
            is_method = False
            for potential_class in ast.walk(root):
                if isinstance(potential_class, ast.ClassDef):
                    for item in potential_class.body:
                        if item is node:
                            is_method = True
                            break

            functions.append(FunctionInfo(
                name=node.name,
                start_line=start,
                end_line=end,
                parameter_count=param_count,
                body_line_count=body_lines,
                decorators=decorators,
                is_method=is_method,
                docstring=docstring,
            ))

    return functions


def _extract_generic_functions(root: GenericNode) -> List[FunctionInfo]:
    """Extract function definitions from a GenericNode tree."""
    functions: List[FunctionInfo] = []
    for node in _generic_walk(root):
        if node.node_type == "function_def":
            func_name = node.attributes.get("name", "unknown")
            functions.append(FunctionInfo(
                name=func_name,
                start_line=node.start_line,
                end_line=node.end_line,
                body_line_count=node.end_line - node.start_line,
                body_text=node.text,
            ))
    return functions


def _generic_walk(node: GenericNode) -> Generator[GenericNode, None, None]:
    """Walk a GenericNode tree yielding every node."""
    yield node
    for child in node.children:
        yield from _generic_walk(child)


def find_functions(node: Any) -> List[FunctionInfo]:
    """Find all function definitions under *node*.

    Parameters
    ----------
    node : ast.AST | GenericNode
        Root of the tree to search.

    Returns
    -------
    List[FunctionInfo]
        Extracted function metadata.
    """
    if isinstance(node, ast.AST):
        return _extract_python_functions(node)
    if isinstance(node, GenericNode):
        return _extract_generic_functions(node)
    logger.warning("Unsupported node type for find_functions: %s", type(node).__name__)
    return []


# ---------------------------------------------------------------------------
# Loop extraction
# ---------------------------------------------------------------------------

def _extract_python_loops(root: ast.AST) -> List[LoopInfo]:
    """Extract loop constructs from a Python AST with nesting depth."""
    loops: List[LoopInfo] = []

    def _visit(node: ast.AST, depth: int, parent_func: Optional[str]) -> None:
        current_func = parent_func
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current_func = node.name

        if isinstance(node, ast.For):
            target_str = ""
            if isinstance(node.target, ast.Name):
                target_str = node.target.id
            loops.append(LoopInfo(
                keyword="for",
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                nesting_depth=depth,
                parent_function=current_func,
                iteration_target=target_str,
            ))
            for child in ast.iter_child_nodes(node):
                _visit(child, depth + 1, current_func)
            return

        if isinstance(node, ast.While):
            loops.append(LoopInfo(
                keyword="while",
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                nesting_depth=depth,
                parent_function=current_func,
            ))
            for child in ast.iter_child_nodes(node):
                _visit(child, depth + 1, current_func)
            return

        for child in ast.iter_child_nodes(node):
            _visit(child, depth, current_func)

    _visit(root, 1, None)
    return loops


def _extract_generic_loops(root: GenericNode) -> List[LoopInfo]:
    """Extract loops from a GenericNode tree."""
    loops: List[LoopInfo] = []

    def _visit(node: GenericNode, depth: int, parent_func: Optional[str]) -> None:
        current_func = parent_func
        if node.node_type == "function_def":
            current_func = node.attributes.get("name")

        if node.node_type == "loop":
            loops.append(LoopInfo(
                keyword=node.attributes.get("keyword", "for"),
                start_line=node.start_line,
                end_line=node.end_line,
                nesting_depth=depth,
                parent_function=current_func,
            ))
            for child in node.children:
                _visit(child, depth + 1, current_func)
            return

        for child in node.children:
            _visit(child, depth, current_func)

    _visit(root, 1, None)
    return loops


def find_loops(node: Any) -> List[LoopInfo]:
    """Find all loop constructs under *node*.

    Parameters
    ----------
    node : ast.AST | GenericNode
        Root of the tree to search.

    Returns
    -------
    List[LoopInfo]
        Extracted loop metadata with nesting depths.
    """
    if isinstance(node, ast.AST):
        return _extract_python_loops(node)
    if isinstance(node, GenericNode):
        return _extract_generic_loops(node)
    logger.warning("Unsupported node type for find_loops: %s", type(node).__name__)
    return []


# ---------------------------------------------------------------------------
# Condition extraction
# ---------------------------------------------------------------------------

def _extract_python_conditions(root: ast.AST) -> List[ConditionInfo]:
    """Extract conditional branches from a Python AST."""
    conditions: List[ConditionInfo] = []

    for node in ast.walk(root):
        if isinstance(node, ast.If):
            has_else = bool(node.orelse)
            # Count branches: 1 for the if, plus elif/else
            branch_count = 1
            current: Any = node
            while current.orelse:
                branch_count += 1
                if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    current = current.orelse[0]
                else:
                    break

            # Determine parent function
            parent_func: Optional[str] = None
            for fn_node in ast.walk(root):
                if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for descendant in ast.walk(fn_node):
                        if descendant is node:
                            parent_func = fn_node.name
                            break

            conditions.append(ConditionInfo(
                keyword="if",
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                has_else=has_else,
                branch_count=branch_count,
                parent_function=parent_func,
            ))

    return conditions


def _extract_generic_conditions(root: GenericNode) -> List[ConditionInfo]:
    """Extract conditions from a GenericNode tree."""
    conditions: List[ConditionInfo] = []
    for node in _generic_walk(root):
        if node.node_type == "condition":
            keyword = node.attributes.get("keyword", "if")
            parent_func: Optional[str] = None
            # Simple parent detection
            for fn in _generic_walk(root):
                if fn.node_type == "function_def" and node in fn.children:
                    parent_func = fn.attributes.get("name")
                    break
            conditions.append(ConditionInfo(
                keyword=keyword,
                start_line=node.start_line,
                end_line=node.end_line,
                parent_function=parent_func,
            ))
    return conditions


def find_conditions(node: Any) -> List[ConditionInfo]:
    """Find all conditional constructs under *node*.

    Parameters
    ----------
    node : ast.AST | GenericNode
        Root of the tree to search.

    Returns
    -------
    List[ConditionInfo]
        Extracted condition metadata.
    """
    if isinstance(node, ast.AST):
        return _extract_python_conditions(node)
    if isinstance(node, GenericNode):
        return _extract_generic_conditions(node)
    logger.warning("Unsupported node type for find_conditions: %s", type(node).__name__)
    return []


# ---------------------------------------------------------------------------
# Variable extraction
# ---------------------------------------------------------------------------

def _extract_python_variables(root: ast.AST) -> List[VariableInfo]:
    """Extract variable assignments / declarations from a Python AST."""
    variables: List[VariableInfo] = []
    seen_names: set = set()

    # Build a mapping of node -> parent function for scope detection
    func_map: dict = {}
    for fn_node in ast.walk(root):
        if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for descendant in ast.walk(fn_node):
                func_map[id(descendant)] = fn_node.name

    class_names: set = set()
    for cn in ast.walk(root):
        if isinstance(cn, ast.ClassDef):
            class_names.add(cn.name)

    for node in ast.walk(root):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = _extract_target_names(target)
                for name in names:
                    key = (name, node.lineno)
                    if key in seen_names:
                        continue
                    seen_names.add(key)

                    parent_func = func_map.get(id(node))
                    scope = "local" if parent_func else "global"

                    # Check for type annotation on the target (not present
                    # on plain Assign, but we mark it as None)
                    variables.append(VariableInfo(
                        name=name,
                        line=node.lineno,
                        scope=scope,
                        parent_function=parent_func,
                    ))

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                key = (name, node.lineno)
                if key not in seen_names:
                    seen_names.add(key)
                    parent_func = func_map.get(id(node))
                    scope = "local" if parent_func else "global"
                    type_ann: Optional[str] = None
                    if isinstance(node.annotation, ast.Name):
                        type_ann = node.annotation.id
                    elif isinstance(node.annotation, ast.Constant):
                        type_ann = str(node.annotation.value)
                    variables.append(VariableInfo(
                        name=name,
                        line=node.lineno,
                        scope=scope,
                        type_annotation=type_ann,
                        parent_function=parent_func,
                    ))

    return variables


def _extract_target_names(target: ast.AST) -> List[str]:
    """Recursively extract variable names from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple) or isinstance(target, ast.List):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_extract_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _extract_target_names(target.value)
    return []


def _extract_generic_variables(root: GenericNode) -> List[VariableInfo]:
    """Extract variables from a GenericNode tree."""
    variables: List[VariableInfo] = []
    for node in _generic_walk(root):
        if node.node_type == "variable":
            var_name = node.attributes.get("name", "unknown")
            parent_func: Optional[str] = None
            for fn in _generic_walk(root):
                if fn.node_type == "function_def" and node in fn.children:
                    parent_func = fn.attributes.get("name")
                    break
            scope = "local" if parent_func else "global"
            variables.append(VariableInfo(
                name=var_name,
                line=node.start_line,
                scope=scope,
                parent_function=parent_func,
            ))
    return variables


def find_variables(node: Any) -> List[VariableInfo]:
    """Find all variable declarations / assignments under *node*.

    Parameters
    ----------
    node : ast.AST | GenericNode
        Root of the tree to search.

    Returns
    -------
    List[VariableInfo]
        Extracted variable metadata.
    """
    if isinstance(node, ast.AST):
        return _extract_python_variables(node)
    if isinstance(node, GenericNode):
        return _extract_generic_variables(node)
    logger.warning("Unsupported node type for find_variables: %s", type(node).__name__)
    return []


# ---------------------------------------------------------------------------
# Nesting depth
# ---------------------------------------------------------------------------

def _python_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Compute the maximum nesting depth of control-flow structures in a
    Python AST.
    """
    max_depth = current_depth

    nesting_node_types = (ast.If, ast.For, ast.While, ast.With, ast.Try)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_node_types):
            child_depth = _python_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _python_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)

    return max_depth


def _generic_nesting_depth(node: GenericNode, current_depth: int = 0) -> int:
    """Compute the maximum nesting depth for a GenericNode tree."""
    max_depth = current_depth

    nesting_types = {"loop", "condition"}

    for child in node.children:
        if child.node_type in nesting_types:
            child_depth = _generic_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _generic_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)

    return max_depth


def get_nesting_depth(node: Any) -> int:
    """Compute the maximum nesting depth of control-flow structures.

    Counts nested ``if``, ``for``, ``while``, ``with``, and ``try``
    blocks.

    Parameters
    ----------
    node : ast.AST | GenericNode
        Root node.

    Returns
    -------
    int
        Maximum nesting depth (0 means no nesting).
    """
    if isinstance(node, ast.AST):
        return _python_nesting_depth(node)
    if isinstance(node, GenericNode):
        return _generic_nesting_depth(node)
    logger.warning("Unsupported node type for get_nesting_depth: %s", type(node).__name__)
    return 0
