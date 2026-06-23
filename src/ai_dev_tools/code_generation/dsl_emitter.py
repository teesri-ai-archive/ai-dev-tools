"""
dsl_emitter.py
==============
Render the DSL schema as a Python-like reference block for prompts/docs.
"""

from __future__ import annotations

from ai_dev_tools.code_generation.dsl_schema import (
    DSL_FUNCTIONS,
    DSL_TYPES,
    DSLFunction,
    DSLType,
    resolve_type_spec,
)


def _to_python_type_name(type_spec: object) -> str:
    """Map schema type specs to the prompt-facing Python type notation."""
    resolved = resolve_type_spec(type_spec)  # type: ignore[arg-type]
    if resolved == "Literal[str]":
        return "str"
    if resolved == "Literal[int]":
        return "int"
    if resolved == "Literal[float]":
        return "float"
    return resolved


def _render_docstring_block(text: str) -> list[str]:
    return [
        '  """',
        f"  {text}",
        '  """',
    ]


def _render_type_definition(dsl_type: DSLType) -> list[str]:
    base = f"({dsl_type.parent})" if dsl_type.parent else ""
    lines: list[str] = [f"class {dsl_type.name}{base}:"]

    if dsl_type.javadoc:
        lines.extend(_render_docstring_block(dsl_type.javadoc))

    if dsl_type.constructor_params:
        for name, param_type in dsl_type.constructor_params:
            lines.append(f"  {name}: {_to_python_type_name(param_type)}")
        return lines

    lines.append("  ...")
    return lines


def _render_return_type(return_type: object) -> str:
    if return_type is None:
        return "None"
    if isinstance(return_type, list):
        rendered_members = ", ".join(_to_python_type_name(member) for member in return_type)
        return f"tuple[{rendered_members}]"
    return _to_python_type_name(return_type)


def _render_function_definition(function: DSLFunction) -> list[str]:
    rendered_params = ", ".join(
        f"{name}: {_to_python_type_name(param_type)}"
        for name, param_type in function.param_types
    )
    signature = (
        f"def {function.name}({rendered_params}) -> "
        f"{_render_return_type(function.return_type)}:"
    )

    lines: list[str] = [signature]
    if function.javadoc:
        lines.extend(_render_docstring_block(function.javadoc))
    lines.append("  ...")
    return lines


def emit_dsl_reference(
    *,
    dsl_types: dict[str, DSLType] | None = None,
    dsl_functions: dict[str, DSLFunction] | None = None,
) -> str:
    """
    Emit the DSL API surface in the prompt style used by coding_dsl_rules.
    """
    types = DSL_TYPES if dsl_types is None else dsl_types
    functions = DSL_FUNCTIONS if dsl_functions is None else dsl_functions

    rendered_lines: list[str] = []
    for dsl_type in types.values():
        rendered_lines.extend(_render_type_definition(dsl_type))
        rendered_lines.append("")
    for function in functions.values():
        rendered_lines.extend(_render_function_definition(function))
        rendered_lines.append("")

    return "\n".join(rendered_lines).rstrip()
