"""
dsl_parser.py
=============
Parses and *statically* validates DSL source code.

Validation guarantees (nothing reaches eval() unless ALL pass):
  • Top-level statements are assignments, standalone function calls, or simple function defs
  • Simple function defs are zero-arg, non-nested, no return annotation, and must return AudioVideoStream
  • Every RHS is either a Call node, a supported literal constant, or a simple attribute access (x.y)
  • No complex RHS expressions (subscripts, lambdas, chained attrs, etc.)
  • Every called name is a known DSL function OR a constructible DSL type
  • Argument count matches the schema
  • Argument types match the schema (tracked through a symbol table)
  • Literal arguments are of the right Python kind (str/int/float)
  • Literal str arguments pass their registered domain validator (if any)
  • LHS is either a single Name or a Tuple of Names (no nested unpacking)
  • Tuple-return functions are only assigned to a tuple of the right arity
  • Scalar-return functions are only assigned to a single name
  • No name shadows a DSL function or type name
  • "_" is accepted as a wildcard / ignored slot in tuple unpacking

Returns a list of DSLStatement dataclasses that the executor can use directly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from .dsl_schema import (
    DSL_TYPES,
    DSL_FUNCTIONS,
    DSL_LITERAL_VALIDATORS,
    DSLFunction,
    resolve_type_spec,
)

# ---------------------------------------------------------------------------
# Public data structures produced by the parser
# ---------------------------------------------------------------------------

@dataclass
class DSLCall:
    """A validated, resolved call ready for execution."""
    callee: str                    # function or type name
    args: list[Any]                # Python literals or variable name strings
    arg_kinds: list[str]           # "literal" | "variable"
    return_types: list[str]        # resolved return types (may be empty)


@dataclass
class DSLStatement:
    """One validated statement."""
    lhs: list[str]                 # variable names; "_" means discard; empty => standalone call
    call: DSLCall
    lineno: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESERVED = set(DSL_TYPES) | set(DSL_FUNCTIONS)

_LITERAL_KIND_MAP = {
    ast.Constant: None,  # handled specially below
}


def _format_expression(node: ast.AST | None) -> str:
    """Render an AST node as readable source for diagnostics."""
    if node is None:
        return "<unknown>"
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DSLParseError(Exception):
    """
    Base class for parser/validation failures with structured diagnostics.
    """

    def __init__(
        self,
        detail: str,
        *,
        lineno: int | None = None,
        expression: str | None = None,
    ) -> None:
        self.detail = detail
        self.lineno = lineno
        self.expression = expression

        chunks: list[str] = []
        if lineno is not None:
            chunks.append(f"Line {lineno}")
        if expression:
            chunks.append(f"Expression: {expression!r}")
        chunks.append(detail)
        super().__init__(" | ".join(chunks))


class DSLValidationError(DSLParseError):
    """Raised when DSL source code violates validation rules."""


class DSLLiteralParseError(DSLParseError):
    """Raised when a DSL literal cannot be parsed to a supported type."""


def _ast_to_python_literal(node: ast.expr) -> tuple[Any, str]:
    """
    Return (python_value, dsl_literal_type) for a literal AST node.
    dsl_literal_type is one of "Literal[str]", "Literal[int]", "Literal[float]".
    Raises DSLLiteralParseError if the node is not a supported plain literal.
    """
    if not isinstance(node, ast.Constant):
        raise DSLLiteralParseError(
            "Expected a literal value (str, int, or float constant)",
            expression=_format_expression(node),
        )
    v = node.value
    if isinstance(v, bool):  # bool is subclass of int – disallow it
        raise DSLLiteralParseError(
            "Boolean literals are not allowed in the DSL",
            expression=_format_expression(node),
        )
    if isinstance(v, str):
        return v, "Literal[str]"
    if isinstance(v, int):
        return v, "Literal[int]"
    if isinstance(v, float):
        return v, "Literal[float]"
    raise DSLLiteralParseError(
        f"Unsupported literal type {type(v).__name__!r}",
        expression=_format_expression(node),
    )


def _is_subtype(child: str, parent: str) -> bool:
    """
    Return True if *child* is the same as or a subtype of *parent*
    according to the DSL_TYPES inheritance chain.
    """
    if child == parent:
        return True
    t = DSL_TYPES.get(child)
    if t is None or t.parent is None:
        return False
    return _is_subtype(t.parent, parent)


def _check_arg_type(
    actual: str,
    expected: str,
    arg_index: int,
    callee: str,
    *,
    lineno: int | None = None,
    expression: str | None = None,
) -> None:
    """Raise DSLValidationError if actual type doesn't satisfy expected."""
    # Literal types must match exactly
    if expected.startswith("Literal["):
        if actual != expected:
            raise DSLValidationError(
                f"Argument {arg_index} of '{callee}' expects {expected} "
                f"but got {actual!r}",
                lineno=lineno,
                expression=expression,
            )
        return
    # Object types: allow subtype polymorphism
    if not _is_subtype(actual, expected):
        raise DSLValidationError(
            f"Argument {arg_index} of '{callee}' expects type '{expected}' "
            f"but got '{actual}'",
            lineno=lineno,
            expression=expression,
        )


# ---------------------------------------------------------------------------
# Main validator / parser
# ---------------------------------------------------------------------------

class DSLParser:
    def __init__(self) -> None:
        # symbol table: variable name → DSL type name
        self._symbols: dict[str, str] = {}
        # user-defined helper functions declared in the DSL source
        self._user_functions: dict[str, ast.FunctionDef] = {}
        # protects against recursive helper functions during inlining
        self._function_expansion_stack: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, source: str) -> list[DSLStatement]:
        """
        Parse and validate *source*. Returns a list of DSLStatements.
        Raises DSLValidationError on any violation.
        """
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            expression = (exc.text or "").strip() or "<unknown>"
            detail = f"Syntax error: {exc.msg} (column {exc.offset})"
            raise DSLValidationError(
                detail,
                lineno=exc.lineno,
                expression=expression,
            ) from exc

        # Parse() should start from a clean state.
        self._symbols = {}
        self._user_functions = {}
        self._function_expansion_stack = []

        self._collect_user_functions(tree.body)

        statements: list[DSLStatement] = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                continue
            statements.extend(self._validate_statement(node))

        return statements

    # ------------------------------------------------------------------
    # Statement-level validation
    # ------------------------------------------------------------------

    def _raise_validation_error(self, detail: str, *, lineno: int, node: ast.AST | None) -> None:
        raise DSLValidationError(
            detail,
            lineno=lineno,
            expression=_format_expression(node),
        )

    def _validate_statement(self, node: ast.stmt) -> list[DSLStatement]:
        lhs_names: list[str]
        call: DSLCall

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                self._raise_validation_error(
                    "Multiple assignment targets (a=b=...) are not allowed",
                    lineno=node.lineno,
                    node=node,
                )
            lhs_names = self._validate_lhs(node.targets[0], node.lineno)
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in self._user_functions
            ):
                prelude, call = self._expand_user_function_call(
                    function_name=node.value.func.id,
                    call_node=node.value,
                    lineno=node.lineno,
                )
                return prelude + [self._finalize_statement(node=node, lhs_names=lhs_names, call=call)]
            call = self._validate_rhs(node.value, node.lineno)
            return [self._finalize_statement(node=node, lhs_names=lhs_names, call=call)]

        elif isinstance(node, ast.Expr):
            if not isinstance(node.value, ast.Call):
                self._raise_validation_error(
                    "Standalone expressions must be plain function calls",
                    lineno=node.lineno,
                    node=node.value,
                )
            lhs_names = []
            if isinstance(node.value.func, ast.Name) and node.value.func.id in self._user_functions:
                prelude, call = self._expand_user_function_call(
                    function_name=node.value.func.id,
                    call_node=node.value,
                    lineno=node.lineno,
                )
                return prelude + [self._finalize_statement(node=node, lhs_names=lhs_names, call=call)]
            call = self._validate_rhs(node.value, node.lineno)
            return [self._finalize_statement(node=node, lhs_names=lhs_names, call=call)]

        else:
            self._raise_validation_error(
                f"Only assignment statements and function-call expressions are allowed. "
                f"Got {type(node).__name__!r}",
                lineno=node.lineno,
                node=node,
            )

    def _finalize_statement(self, *, node: ast.stmt, lhs_names: list[str], call: DSLCall) -> DSLStatement:
        # ---- arity check between LHS slots and return type ----
        n_returns = len(call.return_types)
        n_lhs = len(lhs_names)

        if n_lhs == 0:
            if n_returns != 0:
                self._raise_validation_error(
                    f"'{call.callee}' returns a value and must be assigned to a variable",
                    lineno=node.lineno,
                    node=node,
                )
            return DSLStatement(lhs=lhs_names, call=call, lineno=node.lineno)

        if n_returns > 1:
            # Tuple return: LHS must be a tuple of exactly the right arity
            if n_lhs == 1 and lhs_names[0] != "_":
                self._raise_validation_error(
                    f"'{call.callee}' returns a tuple of {n_returns} values; "
                    f"LHS must be a tuple of {n_returns} variables",
                    lineno=node.lineno,
                    node=node,
                )
            if n_lhs > 1 and n_lhs != n_returns:
                self._raise_validation_error(
                    f"'{call.callee}' returns {n_returns} values but LHS has {n_lhs} slots",
                    lineno=node.lineno,
                    node=node,
                )
        else:
            # Scalar (or None) return: LHS must be a single name
            if n_lhs > 1:
                self._raise_validation_error(
                    f"'{call.callee}' returns a scalar value but LHS is a tuple",
                    lineno=node.lineno,
                    node=node,
                )

        # ---- update symbol table ----
        if n_returns > 1 and n_lhs == n_returns:
            for name, rtype in zip(lhs_names, call.return_types):
                if name != "_":
                    self._bind(name, rtype, node.lineno)
        elif n_returns == 1:
            if lhs_names[0] != "_":
                self._bind(lhs_names[0], call.return_types[0], node.lineno)
        # n_returns == 0 → None-returning function; we don't bind anything

        return DSLStatement(lhs=lhs_names, call=call, lineno=node.lineno)

    def _collect_user_functions(self, body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.AsyncFunctionDef):
                self._raise_validation_error(
                    "Async function definitions are not allowed in the DSL",
                    lineno=node.lineno,
                    node=node,
                )
            if not isinstance(node, ast.FunctionDef):
                continue

            name = node.name
            self._check_not_reserved(name, node.lineno)
            if name in self._user_functions:
                self._raise_validation_error(
                    f"Duplicate function definition for '{name}'",
                    lineno=node.lineno,
                    node=node,
                )

            if node.decorator_list:
                self._raise_validation_error(
                    "Function decorators are not allowed in the DSL",
                    lineno=node.lineno,
                    node=node,
                )

            args = node.args
            has_args = any(
                [
                    len(args.posonlyargs) > 0,
                    len(args.args) > 0,
                    args.vararg is not None,
                    len(args.kwonlyargs) > 0,
                    args.kwarg is not None,
                    len(args.defaults) > 0,
                    len(args.kw_defaults) > 0,
                ]
            )
            if has_args:
                self._raise_validation_error(
                    "DSL helper functions must not take any arguments",
                    lineno=node.lineno,
                    node=node,
                )

            if node.returns is not None:
                self._raise_validation_error(
                    "DSL helper functions must not declare return type annotations",
                    lineno=node.lineno,
                    node=node.returns,
                )

            if len(node.body) == 0:
                self._raise_validation_error(
                    f"Function '{name}' must contain a return statement",
                    lineno=node.lineno,
                    node=node,
                )

            if not isinstance(node.body[-1], ast.Return):
                self._raise_validation_error(
                    f"Function '{name}' must end with a return statement",
                    lineno=node.lineno,
                    node=node,
                )

            for stmt in node.body[:-1]:
                if isinstance(stmt, ast.Return):
                    self._raise_validation_error(
                        f"Function '{name}' can only return from its final statement",
                        lineno=stmt.lineno,
                        node=stmt,
                    )
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._raise_validation_error(
                        "Nested function definitions are not allowed in the DSL",
                        lineno=stmt.lineno,
                        node=stmt,
                    )

            return_stmt = node.body[-1]
            if isinstance(return_stmt, ast.Return) and return_stmt.value is None:
                self._raise_validation_error(
                    f"Function '{name}' must return an AudioVideoStream value",
                    lineno=return_stmt.lineno,
                    node=return_stmt,
                )

            self._user_functions[name] = node

    def _expand_user_function_call(
        self,
        *,
        function_name: str,
        call_node: ast.Call,
        lineno: int,
    ) -> tuple[list[DSLStatement], DSLCall]:
        if call_node.args or call_node.keywords:
            self._raise_validation_error(
                f"Helper function '{function_name}' does not take arguments",
                lineno=lineno,
                node=call_node,
            )

        if function_name in self._function_expansion_stack:
            self._raise_validation_error(
                f"Recursive helper function calls are not allowed ('{function_name}')",
                lineno=lineno,
                node=call_node,
            )

        function_def = self._user_functions[function_name]
        expanded_statements: list[DSLStatement] = []
        self._function_expansion_stack.append(function_name)
        try:
            for stmt in function_def.body[:-1]:
                expanded_statements.extend(self._validate_statement(stmt))

            return_stmt = function_def.body[-1]
            assert isinstance(return_stmt, ast.Return)
            if return_stmt.value is None:
                self._raise_validation_error(
                    f"Function '{function_name}' must return an AudioVideoStream value",
                    lineno=return_stmt.lineno,
                    node=return_stmt,
                )

            return_call = self._validate_function_return_expr(
                function_name=function_name,
                return_node=return_stmt.value,
                lineno=return_stmt.lineno,
            )
            return expanded_statements, return_call
        finally:
            self._function_expansion_stack.pop()

    def _validate_function_return_expr(
        self,
        *,
        function_name: str,
        return_node: ast.expr,
        lineno: int,
    ) -> DSLCall:
        if (
            isinstance(return_node, ast.Call)
            and isinstance(return_node.func, ast.Name)
            and return_node.func.id in self._user_functions
        ):
            callee = return_node.func.id
            if callee in self._function_expansion_stack:
                self._raise_validation_error(
                    f"Recursive helper function calls are not allowed ('{callee}')",
                    lineno=lineno,
                    node=return_node,
                )
            self._raise_validation_error(
                "Helper functions cannot be called directly in return statements; "
                "assign the result to a variable first",
                lineno=lineno,
                node=return_node,
            )

        if isinstance(return_node, ast.Name):
            var_name = return_node.id
            if var_name not in self._symbols:
                self._raise_validation_error(
                    f"Variable '{var_name}' is used before assignment",
                    lineno=lineno,
                    node=return_node,
                )
            return_type = self._symbols[var_name]
            if not _is_subtype(return_type, "AudioVideoStream"):
                self._raise_validation_error(
                    f"Function '{function_name}' must return 'AudioVideoStream' "
                    f"but got '{return_type}'",
                    lineno=lineno,
                    node=return_node,
                )
            return DSLCall(
                callee="__dsl_identity__",
                args=[var_name],
                arg_kinds=["variable"],
                return_types=[return_type],
            )

        call = self._validate_rhs(return_node, lineno)
        if len(call.return_types) != 1 or not _is_subtype(call.return_types[0], "AudioVideoStream"):
            self._raise_validation_error(
                f"Function '{function_name}' must return 'AudioVideoStream'",
                lineno=lineno,
                node=return_node,
            )
        return call

    # ------------------------------------------------------------------
    # LHS validation
    # ------------------------------------------------------------------

    def _validate_lhs(self, target: ast.expr, lineno: int) -> list[str]:
        if isinstance(target, ast.Name):
            name = target.id
            self._check_not_reserved(name, lineno)
            return [name]

        if isinstance(target, ast.Tuple):
            names: list[str] = []
            for elt in target.elts:
                if not isinstance(elt, ast.Name):
                    self._raise_validation_error(
                        "Nested tuple unpacking is not allowed",
                        lineno=lineno,
                        node=target,
                    )
                self._check_not_reserved(elt.id, lineno)
                names.append(elt.id)
            if len(names) == 0:
                self._raise_validation_error(
                    "Empty tuple on LHS is not allowed",
                    lineno=lineno,
                    node=target,
                )
            return names

        self._raise_validation_error(
            f"Invalid assignment target {type(target).__name__!r}; "
            "only simple names or tuples of names are allowed",
            lineno=lineno,
            node=target,
        )

    # ------------------------------------------------------------------
    # RHS validation
    # ------------------------------------------------------------------

    def _validate_rhs(self, node: ast.expr, lineno: int) -> DSLCall:
        if isinstance(node, ast.Constant):
            value, literal_type = _ast_to_python_literal(node)
            return DSLCall(
                callee="__dsl_identity__",
                args=[value],
                arg_kinds=["literal"],
                return_types=[literal_type],
            )

        if isinstance(node, ast.Attribute):
            return self._validate_attribute_access_rhs(node, lineno)

        if not isinstance(node, ast.Call):
            self._raise_validation_error(
                f"RHS must be a function/constructor call. Got {type(node).__name__!r}",
                lineno=lineno,
                node=node,
            )

        # Callee must be a bare Name (no attribute access)
        if not isinstance(node.func, ast.Name):
            self._raise_validation_error(
                f"Callee must be a plain name (no attribute access). Got {ast.dump(node.func)!r}",
                lineno=lineno,
                node=node.func,
            )

        callee = node.func.id

        # No star-args
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self._raise_validation_error(
                    "Star-unpacking (*args) is not allowed",
                    lineno=lineno,
                    node=arg,
                )

        # Resolve callee to a schema entry
        if callee in DSL_FUNCTIONS:
            schema = DSL_FUNCTIONS[callee]
            return_types = self._resolve_function_return(schema)
            expected_params = schema.param_types
        elif callee in DSL_TYPES and DSL_TYPES[callee].constructible:
            schema_type = DSL_TYPES[callee]
            return_types = [callee]
            expected_params = schema_type.constructor_params
        else:
            self._raise_validation_error(
                f"Unknown function or type {callee!r}",
                lineno=lineno,
                node=node,
            )

        ordered_arg_nodes = self._bind_call_arguments(
            node=node,
            callee=callee,
            expected_params=expected_params,
            lineno=lineno,
        )

        # Validate each argument
        args: list[Any] = []
        arg_kinds: list[str] = []

        for i, (arg_node, (_, expected_type_spec)) in enumerate(zip(ordered_arg_nodes, expected_params)):
            expected_type = resolve_type_spec(expected_type_spec)
            if expected_type.startswith("Literal["):
                # Literal expectations accept either compile-time literals or
                # variables previously bound to literal values.
                if isinstance(arg_node, ast.Name):
                    var_name = arg_node.id
                    if var_name not in self._symbols:
                        self._raise_validation_error(
                            f"Variable '{var_name}' is used before assignment",
                            lineno=lineno,
                            node=arg_node,
                        )
                    actual_lit_type = self._symbols[var_name]
                    value = var_name
                    arg_kind = "variable"
                else:
                    try:
                        value, actual_lit_type = _ast_to_python_literal(arg_node)
                    except DSLLiteralParseError as exc:
                        raise DSLValidationError(
                            f"Argument {i} of '{callee}': {exc.detail}",
                            lineno=lineno,
                            expression=_format_expression(arg_node),
                        ) from exc
                    arg_kind = "literal"

                _check_arg_type(
                    actual_lit_type,
                    expected_type,
                    i,
                    callee,
                    lineno=lineno,
                    expression=_format_expression(arg_node),
                )

                # Domain-specific string validation only for compile-time literals.
                if arg_kind == "literal" and actual_lit_type == "Literal[str]":
                    validators = DSL_LITERAL_VALIDATORS.get(callee, {})
                    validator = validators.get(i)
                    if validator is not None:
                        try:
                            validator(value)
                        except ValueError as exc:
                            raise DSLValidationError(
                                f"Argument {i} of '{callee}' failed validation: {exc}",
                                lineno=lineno,
                                expression=_format_expression(arg_node),
                            ) from exc

                args.append(value)
                arg_kinds.append(arg_kind)

            else:
                # Must be a variable reference
                if not isinstance(arg_node, ast.Name):
                    self._raise_validation_error(
                        f"Argument {i} of '{callee}' must be a variable name "
                        "(no nested expressions)",
                        lineno=lineno,
                        node=arg_node,
                    )
                var_name = arg_node.id
                if var_name not in self._symbols:
                    self._raise_validation_error(
                        f"Variable '{var_name}' is used before assignment",
                        lineno=lineno,
                        node=arg_node,
                    )
                actual_type = self._symbols[var_name]
                _check_arg_type(
                    actual_type,
                    expected_type,
                    i,
                    callee,
                    lineno=lineno,
                    expression=_format_expression(arg_node),
                )

                args.append(var_name)
                arg_kinds.append("variable")

        return DSLCall(
            callee=callee,
            args=args,
            arg_kinds=arg_kinds,
            return_types=return_types,
        )

    def _validate_attribute_access_rhs(self, node: ast.Attribute, lineno: int) -> DSLCall:
        """
        Validate a simple attribute access expression like `var_name.field_name`.
        """
        if not isinstance(node.value, ast.Name):
            self._raise_validation_error(
                "Attribute access base must be a variable name (no nested expressions)",
                lineno=lineno,
                node=node,
            )

        var_name = node.value.id
        if var_name not in self._symbols:
            self._raise_validation_error(
                f"Variable '{var_name}' is used before assignment",
                lineno=lineno,
                node=node.value,
            )

        owner_type = self._symbols[var_name]
        attr_name = node.attr
        attr_type = self._resolve_attribute_type(owner_type, attr_name)
        if attr_type is None:
            self._raise_validation_error(
                f"Type '{owner_type}' has no attribute '{attr_name}' in the DSL",
                lineno=lineno,
                node=node,
            )

        return DSLCall(
            callee="__dsl_get_attr__",
            args=[var_name, attr_name],
            arg_kinds=["variable", "literal"],
            return_types=[attr_type],
        )

    def _bind_call_arguments(
        self,
        *,
        node: ast.Call,
        callee: str,
        expected_params: list[tuple[str, object]],
        lineno: int,
    ) -> list[ast.expr]:
        """
        Normalize positional/keyword call arguments into schema parameter order.
        """
        n_expected = len(expected_params)
        if not node.keywords:
            if len(node.args) != n_expected:
                self._raise_validation_error(
                    f"'{callee}' expects {n_expected} argument(s) but got {len(node.args)}",
                    lineno=lineno,
                    node=node,
                )
            return list(node.args)

        if len(node.args) > n_expected:
            self._raise_validation_error(
                f"'{callee}' expects {n_expected} argument(s) but got {len(node.args)}",
                lineno=lineno,
                node=node,
            )

        keyword_values: dict[str, ast.expr] = {}
        for kw in node.keywords:
            if kw.arg is None:
                self._raise_validation_error(
                    "Dictionary-unpacking (**kwargs) is not allowed",
                    lineno=lineno,
                    node=kw,
                )
            if kw.arg in keyword_values:
                self._raise_validation_error(
                    f"Duplicate keyword argument '{kw.arg}'",
                    lineno=lineno,
                    node=kw,
                )
            keyword_values[kw.arg] = kw.value

        expected_names = [name for name, _ in expected_params]
        unknown_keywords = sorted(k for k in keyword_values if k not in expected_names)
        if unknown_keywords:
            rendered = ", ".join(repr(name) for name in unknown_keywords)
            self._raise_validation_error(
                f"Unexpected keyword argument(s) for '{callee}': {rendered}",
                lineno=lineno,
                node=node,
            )

        ordered_nodes: list[ast.expr] = []
        missing_params: list[str] = []
        for index, (param_name, _) in enumerate(expected_params):
            if index < len(node.args):
                if param_name in keyword_values:
                    self._raise_validation_error(
                        f"Argument '{param_name}' for '{callee}' is provided both positionally and by keyword",
                        lineno=lineno,
                        node=node,
                    )
                ordered_nodes.append(node.args[index])
                continue

            kw_value = keyword_values.get(param_name)
            if kw_value is None:
                missing_params.append(param_name)
            else:
                ordered_nodes.append(kw_value)

        if missing_params:
            rendered = ", ".join(missing_params)
            self._raise_validation_error(
                f"Missing required argument(s) for '{callee}': {rendered}",
                lineno=lineno,
                node=node,
            )

        return ordered_nodes

    def _resolve_attribute_type(self, owner_type: str, attr_name: str) -> str | None:
        """
        Resolve DSL attribute type from type metadata.
        Attributes are sourced from constructor parameter metadata on the type
        or any of its parent types.
        """
        current = DSL_TYPES.get(owner_type)
        while current is not None:
            for param_name, param_type in current.constructor_params:
                if param_name == attr_name:
                    return resolve_type_spec(param_type)
            if current.parent is None:
                break
            current = DSL_TYPES.get(current.parent)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_function_return(self, schema: DSLFunction) -> list[str]:
        rt = schema.return_type
        if rt is None:
            return []
        if isinstance(rt, list):
            return [resolve_type_spec(t) for t in rt]
        return [resolve_type_spec(rt)]

    def _check_not_reserved(self, name: str, lineno: int) -> None:
        if name in _RESERVED:
            raise DSLValidationError(
                f"'{name}' is a reserved DSL name and cannot be used as a variable",
                lineno=lineno,
                expression=name,
            )

    def _bind(self, name: str, rtype: str, lineno: int) -> None:
        # Allow rebinding (Python semantics); could make this stricter if desired
        self._symbols[name] = rtype


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

def parse_and_validate(source: str) -> list[DSLStatement]:
    """Parse *source*, validate it, and return the statement list."""
    return DSLParser().parse(source)
