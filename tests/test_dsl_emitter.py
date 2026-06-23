from __future__ import annotations

from ai_dev_tools.code_generation.dsl_schema import DSLFunction, DSLType
from ai_dev_tools.code_generation.dsl_emitter import emit_dsl_reference


def test_emit_dsl_reference_renders_expected_prompt_shape() -> None:
    rendered = emit_dsl_reference()

    assert "class AudioVideoStream:" in rendered
    assert "  file_id: str" in rendered
    assert "class ApproximatePointInTime(PointInTime):" in rendered
    assert (
        "def deserialize_point_in_time(serialized_point_in_time: str) -> ApproximatePointInTime:"
        in rendered
    )
    assert (
        "def break_avstream_into_two(avstream: AudioVideoStream, point_in_time: PointInTime) "
        "-> tuple[AudioVideoStream, AudioVideoStream]:"
        in rendered
    )
    assert '  """' in rendered
    assert "  ..." in rendered
    assert "  pass" not in rendered


def test_emit_dsl_reference_supports_custom_schema_inputs() -> None:
    rendered = emit_dsl_reference(
        dsl_types={
            "CustomBase": DSLType(name="CustomBase", javadoc="", constructible=False),
            "CustomLeaf": DSLType(
                name="CustomLeaf",
                javadoc="Leaf type docs.",
                parent="CustomBase",
                constructor_params=[("count", "Literal[int]"), ("ratio", "Literal[float]")],
            ),
        },
        dsl_functions={
            "combine_custom": DSLFunction(
                name="combine_custom",
                javadoc="",
                param_types=[("left", "CustomLeaf"), ("label", "Literal[str]")],
                return_type=["CustomLeaf", "CustomLeaf"],
            )
        },
    )

    assert "class CustomBase:" in rendered
    assert "class CustomBase:\n  ..." in rendered
    assert "class CustomLeaf(CustomBase):" in rendered
    assert "  count: int" in rendered
    assert "  ratio: float" in rendered
    assert (
        "def combine_custom(left: CustomLeaf, label: str) -> tuple[CustomLeaf, CustomLeaf]:"
        in rendered
    )
    assert "  ..." in rendered
    assert "  pass" not in rendered


def test_emit_dsl_reference_empty_input_returns_empty_string() -> None:
    assert emit_dsl_reference(dsl_types={}, dsl_functions={}) == ""
