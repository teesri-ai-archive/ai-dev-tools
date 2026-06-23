"""
dsl_executor.py
===============
Executes a list of validated DSLStatements produced by dsl_parser.parse().

Design principles
-----------------
* **No eval() of user strings.**  The executor interprets the DSLStatement
  dataclasses directly.  The only Python code that runs is code that was
  already in dsl_builtins.py before the DSL program was received.
* A fresh `env` dict is created for every execution run (no shared mutable
  state between runs).
* The only names that can be looked up in `env` are names that the parser
  already verified as valid assignment targets.
"""

from __future__ import annotations

from typing import Any

from .dsl_parser import DSLCall, DSLStatement
from .dsl_builtins import BUILTIN_IMPLEMENTATIONS


class DSLExecutionError(Exception):
    """Raised when a DSL built-in raises an unexpected exception at runtime."""


def execute(statements: list[DSLStatement]) -> dict[str, Any]:
    """
    Execute *statements* sequentially.

    Returns the final environment (variable name → value) so callers can
    inspect results in tests.
    """
    env: dict[str, Any] = {}

    for stmt in statements:
        result = _execute_call(stmt.call, env, stmt.lineno)

        lhs = stmt.lhs
        n_lhs = len(lhs)
        n_returns = len(stmt.call.return_types)

        if n_lhs == 0:
            # Standalone call statement; parser guarantees this is no-return.
            continue

        if n_returns > 1:
            # Tuple return – result must be iterable with exactly n_returns items
            if n_lhs == n_returns:
                for name, value in zip(lhs, result):
                    if name != "_":
                        env[name] = value
            # If n_lhs == 1 it's a wildcard "_" or the whole tuple kept together –
            # the parser already validated this case; store the tuple
            elif n_lhs == 1 and lhs[0] == "_":
                pass  # discard entirely
        elif n_returns == 1:
            if lhs[0] != "_":
                env[lhs[0]] = result
        # n_returns == 0 → None return (e.g. emit_output); nothing to store

    return env


def _execute_call(call: DSLCall, env: dict[str, Any], lineno: int) -> Any:
    """Resolve arguments and invoke the builtin callable."""
    # Look up the callable – guaranteed to exist because the parser validated it
    callee = BUILTIN_IMPLEMENTATIONS[call.callee]

    # Resolve arguments
    resolved_args: list[Any] = []
    for arg, kind in zip(call.args, call.arg_kinds):
        if kind == "literal":
            resolved_args.append(arg)
        else:  # "variable"
            if arg not in env:
                raise DSLExecutionError(
                    f"Line {lineno}: Variable '{arg}' is not defined at runtime "
                    "(this should have been caught by the validator)"
                )
            resolved_args.append(env[arg])

    # Invoke
    try:
        return callee(*resolved_args)
    except Exception as exc:
        raise DSLExecutionError(
            f"Line {lineno}: Error calling '{call.callee}': {exc}"
        ) from exc
