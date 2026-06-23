"""
test_dsl.py
===========
Tests for the DSL parser, validator, and executor.
Run with:  python test_dsl.py
"""

from ai_dev_tools.code_generation.dsl_parser import parse_and_validate, DSLValidationError
from ai_dev_tools.code_generation.dsl_executor import execute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

_results: list[tuple[str, bool, str]] = []


def _test(label: str, ok: bool, detail: str = "") -> None:
    _results.append((label, ok, detail))
    icon = PASS if ok else FAIL
    print(f"  {icon}  {label}" + (f"\n       {detail}" if detail and not ok else ""))


def expect_valid(label: str, source: str) -> dict:
    try:
        stmts = parse_and_validate(source)
        env = execute(stmts)
        _test(label, True)
        return env
    except Exception as exc:
        _test(label, False, str(exc))
        return {}


def expect_parse_valid(label: str, source: str) -> None:
    try:
        parse_and_validate(source)
        _test(label, True)
    except Exception as exc:
        _test(label, False, str(exc))


def expect_invalid(label: str, source: str, fragment: str = "") -> None:
    try:
        stmts = parse_and_validate(source)
        execute(stmts)
        _test(label, False, "Expected an error but none was raised")
    except DSLValidationError as exc:
        msg = str(exc)
        ok = (fragment in msg) if fragment else True
        _test(label, ok, msg if not ok else "")
    except Exception as exc:
        _test(label, False, f"Wrong exception type {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_programs():
    print("\n── Valid programs ──────────────────────────────────────────────")

    env = expect_valid(
        "Construct an AudioVideoStream",
        'stream = AudioVideoStream("file_abc")',
    )
    assert env.get("stream") is not None

    env = expect_valid(
        "Construct an AudioVideoStream with keyword argument",
        'stream = AudioVideoStream(file_id="file_kw")',
    )
    assert env.get("stream") is not None

    env = expect_valid(
        "Deserialize a point in time",
        'pit = deserialize_point_in_time("00:01:30")',
    )
    assert env.get("pit") is not None

    env = expect_valid(
        "Deserialize a point in time with keyword argument",
        'pit = deserialize_point_in_time(serialized_point_in_time="00:01:31")',
    )
    assert env.get("pit") is not None

    env = expect_valid(
        "Deserialize a time range",
        'tr = deserialize_time_range("00:00:00--00:01:30")',
    )
    assert env.get("tr") is not None

    env = expect_valid(
        "Chain of operations",
        """
stream = AudioVideoStream("vid1")
pit_approx = deserialize_point_in_time("00:00:30")
pit_precise = make_precise_point_in_time(pit_approx)
part1, part2 = break_avstream_into_two(stream, pit_precise)
result = concatenate_avstreams(part1, part2)
_ = emit_output(result)
""".strip(),
    )

    expect_valid(
        "Tuple with wildcard _ discard",
        """
stream = AudioVideoStream("src")
pit = deserialize_point_in_time("00:00:10")
part1, _ = break_avstream_into_two(stream, pit)
_ = emit_output(part1)
""".strip(),
    )

    expect_valid(
        "Overlay two streams",
        """
s1 = AudioVideoStream("video")
s2 = AudioVideoStream("audio_only")
combined = overlay_avstreams(s1, s2)
_ = emit_output(combined)
""".strip(),
    )

    expect_valid(
        "PointInTime is accepted where PointInTime expected (ApproximatePointInTime is subtype)",
        """
stream = AudioVideoStream("v")
pit = deserialize_point_in_time("00:01:00")
a, b = break_avstream_into_two(stream, pit)
_ = emit_output(a)
""".strip(),
    )

    expect_valid(
        "Constant assignment is allowed for literal reuse",
        """
time_range_str = "00:00:00--00:00:10"
tr = deserialize_time_range(time_range_str)
start = tr.start
""".strip(),
    )

    expect_valid(
        "Standalone function-call statement is allowed",
        """
stream = AudioVideoStream("v")
emit_output(stream)
""".strip(),
    )

    env = expect_valid(
        "Attribute access from TimeRange values",
        """
tr = deserialize_time_range("start-ish--end-ish")
start_approx = tr.start
end_approx = tr.end
start_pt = make_precise_point_in_time(start_approx)
end_pt = make_precise_point_in_time(end_approx)
""".strip(),
    )
    assert env.get("start_pt") is not None
    assert env.get("end_pt") is not None


def test_sandbox_violations():
    print("\n── Sandbox / structure violations ──────────────────────────────")

    expect_invalid(
        "import statement rejected",
        "import os",
        "Only assignment",
    )

    expect_invalid(
        "exec() call rejected (unknown callee)",
        'x = exec("__import__(\'os\')")',
        "Unknown function",
    )

    expect_invalid(
        "Attribute access rejected",
        'x = os.system("ls")',
        "plain name",
    )

    expect_invalid(
        "Nested expression rejected",
        'x = concatenate_avstreams(AudioVideoStream("a"), AudioVideoStream("b"))',
        "variable name",
    )

    expect_invalid(
        "Unknown keyword argument rejected",
        'stream = AudioVideoStream(unknown="abc")',
        "Unexpected keyword argument",
    )

    expect_invalid(
        "Standalone expression returning a value must be assigned",
        'AudioVideoStream("x")',
        "must be assigned",
    )

    expect_invalid(
        "helper function with args is rejected",
        """
def foo(x):
    return AudioVideoStream("x")
result = foo()
""".strip(),
        "must not take any arguments",
    )

    expect_invalid(
        "class statement rejected",
        "class Foo: pass",
        "Only assignment",
    )

    expect_invalid(
        "for loop rejected",
        "for x in []: pass",
        "Only assignment",
    )

    expect_invalid(
        "Lambda rejected",
        "f = lambda: None",
        "RHS must be a function",
    )

    expect_invalid(
        "Subscript rejected",
        "x = some_list[0]",
        "RHS must be a function",
    )

    expect_invalid(
        "Unknown attribute rejected",
        """
tr = deserialize_time_range("a--b")
x = tr.middle
""".strip(),
        "has no attribute",
    )

    expect_invalid(
        "Nested attribute chain rejected",
        """
tr = deserialize_time_range("a--b")
x = tr.start.raw
""".strip(),
        "Attribute access base must be a variable name",
    )

    expect_invalid(
        "Star unpacking rejected",
        "x = concatenate_avstreams(*args)",
        "Star-unpacking",
    )

    expect_invalid(
        "Chained assignment rejected",
        'a = b = AudioVideoStream("x")',
        "Multiple assignment targets",
    )

    expect_invalid(
        "Augmented assignment rejected",
        'x += AudioVideoStream("a")',
        "Only assignment",
    )

    expect_invalid(
        "Boolean literal rejected",
        "x = AudioVideoStream(True)",
        "Boolean",
    )

    expect_invalid(
        "Walrus operator rejected",
        "(x := AudioVideoStream('a'))",
        "Only assignment",
    )


def test_type_errors():
    print("\n── Type errors ──────────────────────────────────────────────────")

    expect_invalid(
        "Wrong type: str where AudioVideoStream expected",
        """
pit = deserialize_point_in_time("00:01:00")
x = make_precise_point_in_time(pit)
result = concatenate_avstreams(x, x)
""".strip(),
        "expects type 'AudioVideoStream'",
    )

    expect_invalid(
        "Wrong argument count",
        'stream = AudioVideoStream("a", "b")',
        "expects 1 argument",
    )

    expect_invalid(
        "Missing required keyword argument",
        "stream = AudioVideoStream()",
        "expects 1 argument",
    )

    expect_invalid(
        "Use before assignment",
        "stream = concatenate_avstreams(undefined_var, undefined_var)",
        "used before assignment",
    )

    expect_invalid(
        "Scalar return assigned to tuple",
        """
s = AudioVideoStream("x")
a, b = concatenate_avstreams(s, s)
""".strip(),
        "returns a scalar",
    )

    expect_invalid(
        "Tuple return assigned to scalar (without _)",
        """
stream = AudioVideoStream("v")
pit = deserialize_point_in_time("00:00:05")
x = break_avstream_into_two(stream, pit)
""".strip(),
        "returns a tuple",
    )

    expect_invalid(
        "Tuple arity mismatch",
        """
stream = AudioVideoStream("v")
pit = deserialize_point_in_time("00:00:05")
a, b, c = break_avstream_into_two(stream, pit)
""".strip(),
        "2 values but LHS has 3",
    )

    expect_invalid(
        "Shadowing a reserved DSL name",
        'AudioVideoStream = AudioVideoStream("x")',
        "reserved",
    )


def test_literal_validation():
    print("\n── Literal / domain validators ─────────────────────────────────")

    expect_parse_valid(
        "Arbitrary point-in-time string is accepted by parser validators",
        'pit = deserialize_point_in_time("not-a-time")',
    )

    expect_parse_valid(
        "Arbitrary time-range string without separator is accepted by parser validators",
        'tr = deserialize_time_range("00:00:00")',
    )

    expect_parse_valid(
        "Arbitrary time-range string with non-time parts is accepted by parser validators",
        'tr = deserialize_time_range("00:00:00--badtime")',
    )

    expect_valid(
        "Valid time with milliseconds",
        'pit = deserialize_point_in_time("01:23:45.678")',
    )

    expect_valid(
        "Valid time range with milliseconds",
        'tr = deserialize_time_range("00:00:00.000--01:23:45.678")',
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_valid_programs()
    test_sandbox_violations()
    test_type_errors()
    test_literal_validation()

    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'─'*60}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        failed_names = [label for label, ok, _ in _results if not ok]
        for name in failed_names:
            print(f"  {FAIL} {name}")
    else:
        print("  — all passed ✓")
