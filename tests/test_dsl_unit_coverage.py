from __future__ import annotations

import ast
from typing import Literal

import pytest

from ai_dev_tools.code_generation import dsl_parser
from ai_dev_tools.code_generation.dsl_builtins import (
    ApproximatePointInTime,
    BUILTIN_IMPLEMENTATIONS,
    PointInTime,
    TimeRange,
)
from ai_dev_tools.code_generation.dsl_executor import (
    DSLExecutionError,
    _execute_call,
    execute,
)
from ai_dev_tools.code_generation.dsl_parser import (
    DSLCall,
    DSLLiteralParseError,
    DSLParseError,
    DSLParser,
    DSLStatement,
    DSLValidationError,
    _ast_to_python_literal,
    _check_arg_type,
    parse_and_validate,
)
from ai_dev_tools.code_generation.dsl_schema import DSLFunction, DSLType, resolve_type_spec


def test_resolve_type_spec_supports_all_documented_forms() -> None:
    assert resolve_type_spec("AudioVideoStream") == "AudioVideoStream"
    assert resolve_type_spec("Literal[str]") == "Literal[str]"
    assert resolve_type_spec(DSLType(name="CustomType", javadoc="")) == "CustomType"
    assert resolve_type_spec(eval("Literal[str]")) == "Literal[str]"
    assert resolve_type_spec(eval("Literal[int]")) == "Literal[int]"
    assert resolve_type_spec(eval("Literal[float]")) == "Literal[float]"


def test_resolve_type_spec_rejects_invalid_specs() -> None:
    with pytest.raises(TypeError, match="exactly one type argument"):
        resolve_type_spec(Literal[1, 2])  # pyright: ignore[reportInvalidTypeForm]

    with pytest.raises(TypeError, match="Unsupported Literal type spec"):
        resolve_type_spec(Literal[True])  # pyright: ignore[reportInvalidTypeForm]

    with pytest.raises(TypeError, match="Unsupported DSL type spec"):
        resolve_type_spec(123)  # type: ignore[arg-type]


def test_ast_to_python_literal_uncovered_paths() -> None:
    with pytest.raises(DSLParseError, match="Expected a literal value"):
        _ast_to_python_literal(ast.Name(id="x", ctx=ast.Load()))

    assert _ast_to_python_literal(ast.Constant(value=7)) == (7, "Literal[int]")
    assert _ast_to_python_literal(ast.Constant(value=1.5)) == (1.5, "Literal[float]")

    with pytest.raises(DSLParseError, match="Unsupported literal type"):
        _ast_to_python_literal(ast.Constant(value=None))


def test_check_arg_type_literal_mismatch() -> None:
    with pytest.raises(DSLValidationError, match="expects Literal\\[int\\]"):
        _check_arg_type("Literal[str]", "Literal[int]", 0, "f")


@pytest.mark.parametrize(
    ("source", "message_fragment"),
    [
        (
            'stream = AudioVideoStream(123)',
            r"Argument 0 of 'AudioVideoStream' expects Literal\[str\] but got 'Literal\[int\]'",
        ),
        (
            "stream = AudioVideoStream(deserialize_point_in_time)",
            "Variable 'deserialize_point_in_time' is used before assignment",
        ),
    ],
)
def test_parser_reports_clear_errors_for_literal_type_mismatch_and_function_value(
    source: str, message_fragment: str
) -> None:
    with pytest.raises(DSLValidationError, match=message_fragment) as exc_info:
        parse_and_validate(source)
    assert exc_info.value.lineno == 1
    assert exc_info.value.expression is not None


@pytest.mark.parametrize(
    ("source", "message_fragment"),
    [
        (
            """
s = AudioVideoStream("a")
result = concatenate_avstreams(s)
""".strip(),
            r"'concatenate_avstreams' expects 2 argument\(s\) but got 1",
        ),
        (
            """
s = AudioVideoStream("a")
result = concatenate_avstreams(s, s, s)
""".strip(),
            r"'concatenate_avstreams' expects 2 argument\(s\) but got 3",
        ),
    ],
)
def test_parser_reports_clear_errors_for_too_few_and_too_many_arguments(
    source: str, message_fragment: str
) -> None:
    with pytest.raises(DSLValidationError, match=message_fragment) as exc_info:
        parse_and_validate(source)
    assert exc_info.value.lineno == 2
    assert "concatenate_avstreams" in (exc_info.value.expression or "")


@pytest.mark.parametrize(
    ("source", "message_fragment", "expected_expression"),
    [
        (
            """
pit = deserialize_point_in_time("00:00:10")
s1, s2 = break_avstream_into_two(pit, pit)
""".strip(),
            "expects type 'AudioVideoStream' but got 'ApproximatePointInTime'",
            "pit",
        ),
        (
            """
s = AudioVideoStream("a")
p = make_precise_point_in_time(s)
""".strip(),
            "expects type 'PointInTime' but got 'AudioVideoStream'",
            "s",
        ),
    ],
)
def test_parser_reports_clear_errors_for_incorrect_dsl_types(
    source: str, message_fragment: str, expected_expression: str
) -> None:
    with pytest.raises(DSLValidationError, match=message_fragment) as exc_info:
        parse_and_validate(source)
    assert exc_info.value.lineno == 2
    assert exc_info.value.expression == expected_expression


def test_parser_syntax_error_path() -> None:
    with pytest.raises(DSLValidationError, match="Syntax error") as exc_info:
        DSLParser().parse("x = ")
    exc = exc_info.value
    assert exc.lineno == 1
    assert exc.expression == "x ="
    assert "Line 1" in str(exc)
    assert "Expression:" in str(exc)


def test_parser_uncovered_lhs_validation_paths() -> None:
    parser = DSLParser()

    with pytest.raises(DSLValidationError, match="Nested tuple unpacking") as exc_info:
        parser._validate_lhs(
            ast.Tuple(
                elts=[
                    ast.Name(id="a", ctx=ast.Store()),
                    ast.Tuple(elts=[ast.Name(id="b", ctx=ast.Store())], ctx=ast.Store()),
                ],
                ctx=ast.Store(),
            ),
            lineno=1,
        )
    assert exc_info.value.lineno == 1
    assert "a, (b,)" in (exc_info.value.expression or "")

    with pytest.raises(DSLValidationError, match="Empty tuple on LHS") as exc_info:
        parser._validate_lhs(ast.Tuple(elts=[], ctx=ast.Store()), lineno=1)
    assert exc_info.value.lineno == 1
    assert exc_info.value.expression == "()"

    with pytest.raises(DSLValidationError, match="Invalid assignment target") as exc_info:
        parser._validate_lhs(ast.Constant(value=1), lineno=1)
    assert exc_info.value.lineno == 1
    assert exc_info.value.expression == "1"


def test_parser_literal_non_string_path_without_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        dsl_parser.DSL_FUNCTIONS,
        "expect_int",
        DSLFunction(name="expect_int", javadoc="", param_types=[("x", "Literal[int]")], return_type=None),
    )
    parse_and_validate("x = expect_int(42)")


def test_parser_accepts_positional_and_keyword_arguments() -> None:
    positional = parse_and_validate('stream = AudioVideoStream("positional_file")')
    assert len(positional) == 1
    assert positional[0].call.args == ["positional_file"]
    assert positional[0].call.arg_kinds == ["literal"]

    keyword = parse_and_validate('stream = AudioVideoStream(file_id="keyword_file")')
    assert len(keyword) == 1
    assert keyword[0].call.args == ["keyword_file"]
    assert keyword[0].call.arg_kinds == ["literal"]


def test_parser_accepts_simple_attribute_access() -> None:
    statements = parse_and_validate(
        """
tr = deserialize_time_range("left--right")
start = tr.start
""".strip()
    )
    assert len(statements) == 2
    attr_stmt = statements[1]
    assert attr_stmt.call.callee == "__dsl_get_attr__"
    assert attr_stmt.call.args == ["tr", "start"]
    assert attr_stmt.call.arg_kinds == ["variable", "literal"]
    assert attr_stmt.call.return_types == ["PointInTime"]


def test_parser_accepts_standalone_function_call_statement() -> None:
    statements = parse_and_validate(
        """
stream = AudioVideoStream("left")
emit_output(stream)
""".strip()
    )
    assert len(statements) == 2
    call_stmt = statements[1]
    assert call_stmt.lhs == []
    assert call_stmt.call.callee == "emit_output"
    assert call_stmt.call.args == ["stream"]
    assert call_stmt.call.arg_kinds == ["variable"]
    assert call_stmt.call.return_types == []


def test_parser_accepts_blank_lines_and_python_comments() -> None:
    statements = parse_and_validate(
        """
# this is a DSL comment

stream = AudioVideoStream("left")

# another comment
emit_output(stream)
""".strip()
    )
    assert len(statements) == 2
    assert statements[0].call.callee == "AudioVideoStream"
    assert statements[1].call.callee == "emit_output"


def test_parser_accepts_simple_zero_arg_helper_function_returning_avstream() -> None:
    statements = parse_and_validate(
        """
def build_stream():
    stream = AudioVideoStream("clip")
    return stream

result = build_stream()
emit_output(result)
""".strip()
    )

    # Inlined helper body assignment + inlined return + emit.
    assert len(statements) == 3
    assert [stmt.call.callee for stmt in statements] == [
        "AudioVideoStream",
        "__dsl_identity__",
        "emit_output",
    ]
    assert statements[1].lhs == ["result"]


@pytest.mark.parametrize(
    ("source", "message_fragment"),
    [
        (
            """
def build_stream(x):
    return AudioVideoStream("clip")

result = build_stream()
""".strip(),
            "must not take any arguments",
        ),
        (
            """
def build_stream() -> AudioVideoStream:
    return AudioVideoStream("clip")

result = build_stream()
""".strip(),
            "must not declare return type annotations",
        ),
        (
            """
def build_stream():
    pit = deserialize_point_in_time("00:00:01")
    return pit

result = build_stream()
""".strip(),
            "must return 'AudioVideoStream'",
        ),
        (
            """
def outer():
    def inner():
        return AudioVideoStream("clip")
    return AudioVideoStream("clip")

result = outer()
""".strip(),
            "Nested function definitions are not allowed",
        ),
    ],
)
def test_parser_rejects_invalid_helper_function_shapes(source: str, message_fragment: str) -> None:
    with pytest.raises(DSLValidationError, match=message_fragment):
        parse_and_validate(source)


def test_parser_rejects_recursive_helper_function_calls() -> None:
    with pytest.raises(DSLValidationError, match="Recursive helper function calls are not allowed"):
        parse_and_validate(
            """
def loop():
    return loop()

result = loop()
""".strip()
        )


def test_parser_accepts_constant_assignment_and_literal_variable_usage() -> None:
    statements = parse_and_validate(
        """
time_range_str = "00:00:01--00:00:02"
tr = deserialize_time_range(time_range_str)
""".strip()
    )
    assert len(statements) == 2

    const_stmt = statements[0]
    assert const_stmt.lhs == ["time_range_str"]
    assert const_stmt.call.callee == "__dsl_identity__"
    assert const_stmt.call.args == ["00:00:01--00:00:02"]
    assert const_stmt.call.arg_kinds == ["literal"]
    assert const_stmt.call.return_types == ["Literal[str]"]

    call_stmt = statements[1]
    assert call_stmt.call.callee == "deserialize_time_range"
    assert call_stmt.call.args == ["time_range_str"]
    assert call_stmt.call.arg_kinds == ["variable"]


def test_constant_assignments_preserve_literal_type_for_downstream_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        dsl_parser.DSL_FUNCTIONS,
        "expect_literal_str",
        DSLFunction(
            name="expect_literal_str",
            javadoc="",
            param_types=[("value", "Literal[str]")],
            return_type=None,
        ),
    )
    monkeypatch.setitem(
        dsl_parser.DSL_FUNCTIONS,
        "expect_literal_int",
        DSLFunction(
            name="expect_literal_int",
            javadoc="",
            param_types=[("value", "Literal[int]")],
            return_type=None,
        ),
    )

    parse_and_validate(
        """
literal_str = "abc"
literal_int = 123
expect_literal_str(literal_str)
expect_literal_int(literal_int)
""".strip()
    )

    with pytest.raises(DSLValidationError, match="expects Literal\\[int\\] but got 'Literal\\[str\\]'"):
        parse_and_validate(
            """
literal_str = "abc"
expect_literal_int(literal_str)
""".strip()
        )


def test_variable_redefinition_updates_type_and_runtime_value_from_reassignment_onward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression coverage for Python-like reassignment semantics in the DSL parser.

    A variable keeps its original type/value for earlier statements, can be
    rebound, and from the reassignment onward downstream validation/execution
    must use the new type/value.
    """
    monkeypatch.setitem(
        dsl_parser.DSL_FUNCTIONS,
        "expect_literal_str",
        DSLFunction(
            name="expect_literal_str",
            javadoc="",
            param_types=[("value", "Literal[str]")],
            return_type=None,
        ),
    )
    monkeypatch.setitem(
        dsl_parser.DSL_FUNCTIONS,
        "expect_literal_int",
        DSLFunction(
            name="expect_literal_int",
            javadoc="",
            param_types=[("value", "Literal[int]")],
            return_type=None,
        ),
    )
    monkeypatch.setitem(BUILTIN_IMPLEMENTATIONS, "expect_literal_str", lambda _value: None)
    monkeypatch.setitem(BUILTIN_IMPLEMENTATIONS, "expect_literal_int", lambda _value: None)

    source = """
value = "00:00:01"
expect_literal_str(value)
value = 7
expect_literal_int(value)
""".strip()
    statements = parse_and_validate(source)
    env = execute(statements)
    assert env["value"] == 7

    with pytest.raises(DSLValidationError, match="expects Literal\\[str\\] but got 'Literal\\[int\\]'"):
        parse_and_validate(
            """
value = "00:00:01"
value = 7
expect_literal_str(value)
""".strip()
        )


@pytest.mark.parametrize(
    ("source", "message_fragment"),
    [
        ('stream = AudioVideoStream(unknown="value")', "Unexpected keyword argument"),
        ('stream = AudioVideoStream("a", file_id="b")', "provided both positionally and by keyword"),
        ('stream = AudioVideoStream(**kwargs)', "Dictionary-unpacking"),
        ('tr = deserialize_time_range("a--b")\nx = tr.start.raw', "Attribute access base must be a variable name"),
        ('tr = deserialize_time_range("a--b")\nx = tr.unknown', "has no attribute"),
        ('AudioVideoStream("x")', "must be assigned"),
    ],
)
def test_parser_rejects_invalid_keyword_argument_styles(source: str, message_fragment: str) -> None:
    with pytest.raises(DSLValidationError, match=message_fragment) as exc_info:
        parse_and_validate(source)
    assert exc_info.value.lineno is not None
    assert exc_info.value.expression is not None


@pytest.mark.parametrize(
    ("source", "expected_line", "expected_expression_fragment", "message_fragment"),
    [
        ("import os", 1, "import os", "Only assignment statements and function-call expressions are allowed"),
        ('x = os.system("ls")', 1, "os.system", "plain name"),
        ('stream = AudioVideoStream(unknown="abc")', 1, "AudioVideoStream", "Unexpected keyword argument"),
        ('x = concatenate_avstreams(*args)', 1, "*args", "Star-unpacking"),
        ("x = AudioVideoStream(True)", 1, "True", "Boolean literals"),
        ('AudioVideoStream("x")', 1, "AudioVideoStream", "must be assigned"),
    ],
)
def test_parser_validation_errors_always_include_line_and_expression(
    source: str,
    expected_line: int,
    expected_expression_fragment: str,
    message_fragment: str,
) -> None:
    with pytest.raises(DSLValidationError, match=message_fragment) as exc_info:
        parse_and_validate(source)
    exc = exc_info.value
    assert exc.lineno == expected_line
    assert expected_expression_fragment in (exc.expression or "")
    assert f"Line {expected_line}" in str(exc)
    assert "Expression:" in str(exc)


def test_validation_error_is_a_parse_error_subclass() -> None:
    assert issubclass(DSLValidationError, DSLParseError)


@pytest.mark.parametrize(
    ("source", "detail_fragment", "expression_fragment"),
    [
        # Statement-level validation
            ("import os", "Only assignment statements and function-call expressions are allowed", "import os"),
        ('a = b = AudioVideoStream("x")', "Multiple assignment targets", "a = b ="),
        # LHS validation
        ('AudioVideoStream = AudioVideoStream("x")', "reserved DSL name", "AudioVideoStream"),
        # RHS shape + callee
        ("x = some_list[0]", "RHS must be a function/constructor call", "some_list[0]"),
        ('x = os.system("ls")', "plain name", "os.system"),
        ('stream = AudioVideoStream(unknown="abc")', "Unexpected keyword argument", "AudioVideoStream"),
        ('x = concatenate_avstreams(avstream1=undefined_var, avstream2=undefined_var)', "used before assignment", "undefined_var"),
        ('tr = deserialize_time_range("a--b")\nx = tr.unknown', "has no attribute", "tr.unknown"),
        ("x = concatenate_avstreams(*args)", "Star-unpacking", "*args"),
        ('x = exec("__import__(\\"os\\")")', "Unknown function or type", "exec"),
        ('x = AudioVideoStream("a", "b")', "expects 1 argument", "AudioVideoStream"),
        # Literal failures + variable validation
        ("x = AudioVideoStream(True)", "Boolean literals are not allowed", "True"),
        ("x = AudioVideoStream(None)", "Unsupported literal type", "None"),
        ('x = concatenate_avstreams(AudioVideoStream("a"), AudioVideoStream("b"))', "variable name", "AudioVideoStream('a')"),
        ("x = concatenate_avstreams(undefined_var, undefined_var)", "used before assignment", "undefined_var"),
        # Type mismatch
        (
            """
pit = deserialize_point_in_time("00:01:00")
x = make_precise_point_in_time(pit)
result = concatenate_avstreams(x, x)
""".strip(),
            "expects type 'AudioVideoStream'",
            "x",
        ),
    ],
)
def test_parser_error_matrix_has_structured_metadata(
    source: str, detail_fragment: str, expression_fragment: str
) -> None:
    with pytest.raises(DSLValidationError, match=detail_fragment) as exc_info:
        parse_and_validate(source)
    exc = exc_info.value
    assert exc.lineno is not None
    assert exc.expression is not None
    assert expression_fragment in exc.expression
    rendered = str(exc)
    assert f"Line {exc.lineno}" in rendered
    assert "Expression:" in rendered
    assert detail_fragment in rendered


def test_literal_parse_error_keeps_expression_and_detail() -> None:
    with pytest.raises(DSLLiteralParseError, match="Expected a literal value") as exc_info:
        _ast_to_python_literal(ast.Name(id="x", ctx=ast.Load()))
    exc = exc_info.value
    assert exc.lineno is None
    assert exc.expression == "x"
    assert exc.detail == "Expected a literal value (str, int, or float constant)"
    assert "Expression: 'x'" in str(exc)


def test_parser_scalar_return_discard_does_not_bind_symbol() -> None:
    parser = DSLParser()
    statements = parser.parse('_ = deserialize_point_in_time("00:00:05")')
    assert len(statements) == 1
    assert parser._symbols == {}


def test_executor_tuple_wildcard_discard_branch() -> None:
    statements = parse_and_validate(
        """
stream = AudioVideoStream("vid")
pit = deserialize_point_in_time("00:00:02")
_ = break_avstream_into_two(stream, pit)
""".strip()
    )
    env = execute(statements)
    assert set(env.keys()) == {"stream", "pit"}


def test_execute_call_runtime_undefined_variable_error() -> None:
    call = DSLCall(
        callee="emit_output",
        args=["missing_var"],
        arg_kinds=["variable"],
        return_types=[],
    )
    with pytest.raises(DSLExecutionError, match="not defined at runtime"):
        _execute_call(call, env={}, lineno=1)


def test_execute_call_wraps_builtin_exceptions() -> None:
    call = DSLCall(
        callee="deserialize_time_range",
        args=["bad-range"],
        arg_kinds=["literal"],
        return_types=["TimeRange"],
    )
    with pytest.raises(DSLExecutionError, match="Error calling 'deserialize_time_range'"):
        _execute_call(call, env={}, lineno=1)


def test_time_range_repr_branch() -> None:
    tr = TimeRange(start=ApproximatePointInTime("00:00:01"), end=PointInTime())
    text = repr(tr)
    assert text.startswith("TimeRange(start=")
    assert "end=" in text


def test_execute_scalar_wildcard_discard_statement() -> None:
    statements = [
        DSLStatement(
            lhs=["_"],
            call=DSLCall(
                callee="deserialize_point_in_time",
                args=["00:00:03"],
                arg_kinds=["literal"],
                return_types=["ApproximatePointInTime"],
            ),
            lineno=1,
        )
    ]
    assert execute(statements) == {}


def test_execute_supports_constant_assignment_via_identity_builtin() -> None:
    env = execute(
        parse_and_validate(
            """
time_range_str = "left--right"
tr = deserialize_time_range(time_range_str)
""".strip()
        )
    )
    assert env["time_range_str"] == "left--right"
    assert isinstance(env["tr"], TimeRange)


def test_execute_tuple_return_scalar_lhs_fallback_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Parser rejects this pattern, but executor contains a defensive path.
    monkeypatch.setitem(BUILTIN_IMPLEMENTATIONS, "fake_tuple", lambda: ("left", "right"))
    statements = [
        DSLStatement(
            lhs=["single_name"],
            call=DSLCall(
                callee="fake_tuple",
                args=[],
                arg_kinds=[],
                return_types=["AudioVideoStream", "AudioVideoStream"],
            ),
            lineno=1,
        )
    ]
    env = execute(statements)
    assert env == {}
