"""
dsl_schema.py
=============
The single source of truth for the DSL's type system.

To extend the DSL:
  1. Add your class or function to DSL_TYPES / DSL_FUNCTIONS below.
  2. If a function takes a Literal[str] that needs domain validation, add an
     entry to DSL_LITERAL_VALIDATORS mapping  function_name -> {param_index -> callable}.
  3. Add the real Python implementation to dsl_builtins.py and register it in
     BUILTIN_IMPLEMENTATIONS there.

Nothing else needs to change.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, TypeAlias, get_args, get_origin

# ---------------------------------------------------------------------------
# Type descriptors
# ---------------------------------------------------------------------------

# Keep this alias narrow for static typing (`DSLType | str`) even though
# `resolve_type_spec(...)` also accepts runtime typing objects like `Literal[str]`.
# Reason:
#   1) Pylance/Pyright flags `Literal[str]`/`Literal[int]`/`Literal[float]` in
#      type aliases as invalid `Literal` arguments.
#   2) Most schema entries are authored as strings anyway (e.g. "Literal[str]").
#   3) Runtime flexibility is still preserved via the resolver below.
DSLTypeLike: TypeAlias = "DSLType | str"
DSLParam: TypeAlias = tuple[str, DSLTypeLike]


@dataclass
class DSLType:
    """Describes one type available in the DSL."""
    name: str
    javadoc: str
    # Base type name (None == no parent / is a root type)
    parent: Optional[str] = None
    # True  → can be instantiated with  x = MyType(arg1, arg2, ...)
    # False → abstract / not directly constructible by DSL code
    constructible: bool = True
    # Ordered list of constructor parameters as tuples:
    # (parameter_name, parameter_type).
    #
    # `parameter_type` may be one of:
    #   - a DSL type name string (e.g. "PointInTime"),
    #   - a DSLType object (its .name is used),
    #   - a Python typing literal spec (Literal[str], Literal[int], Literal[float]),
    #   - legacy literal string form ("Literal[str]" etc).
    constructor_params: list[DSLParam] = field(default_factory=list)


@dataclass
class DSLFunction:
    """Describes one function available in the DSL."""
    name: str
    javadoc: str
    # Ordered parameters as tuples: (parameter_name, parameter_type).
    # See DSLType.constructor_params for supported parameter_type forms.
    param_types: list[DSLParam] = field(default_factory=list)
    # Return type as a single type (see supported forms above),
    # a list of types for tuple returns, or None for no return.
    return_type: DSLTypeLike | list[DSLTypeLike] | None = None


def resolve_type_spec(type_spec: DSLTypeLike) -> str:
    """
    Normalize schema type specs to internal canonical strings.

    Supported inputs:
      - "AudioVideoStream" / any DSL type string
      - DSLType(name="AudioVideoStream", ...)
      - Literal[str] / Literal[int] / Literal[float]
      - Legacy strings: "Literal[str]" / "Literal[int]" / "Literal[float]"
    """
    if isinstance(type_spec, DSLType):
        return type_spec.name

    if isinstance(type_spec, str):
        if type_spec in {"Literal[str]", "Literal[int]", "Literal[float]"}:
            return type_spec
        return type_spec

    # This branch is intentionally present even if current in-repo registries
    # mostly use string specs, because callers may provide Pythonic specs
    # (e.g. Literal[str]) in future schema entries. In that case this path is
    # reachable and normalizes the value to the canonical "Literal[...]" string.
    # This path is tested in test_dsl.py.
    origin = get_origin(type_spec)
    if origin is Literal:
        args = get_args(type_spec)
        if len(args) != 1:
            raise TypeError(
                "Literal type spec must have exactly one type argument "
                "(Literal[str], Literal[int], or Literal[float])."
            )
        lit_arg = args[0]
        if lit_arg not in (str, int, float):
            raise TypeError(
                "Unsupported Literal type spec. "
                "Only Literal[str], Literal[int], and Literal[float] are supported."
            )
        return f"Literal[{lit_arg.__name__}]"

    raise TypeError(
        f"Unsupported DSL type spec {type_spec!r}. Expected DSL type name string, "
        "DSLType, or one of Literal[str]/Literal[int]/Literal[float]."
    )


# ---------------------------------------------------------------------------
# DSL Type Registry
# ---------------------------------------------------------------------------

DSL_TYPES: dict[str, DSLType] = {t.name: t for t in [
    DSLType(
        name="AudioVideoStream",
        javadoc="A stream of audio and / or video. (So could be audio only, video only, or audio and video.)",
        constructible=True,
        constructor_params=[("file_id", "Literal[str]")],
    ),
    DSLType(
        name="PointInTime",
        javadoc="",
        constructible=False,   # abstract base
    ),
    DSLType(
        name="ApproximatePointInTime",
        javadoc="",
        parent="PointInTime",
        constructible=False,   # only produced by deserialize_point_in_time
    ),
    DSLType(
        name="PrecisePointInTime",
        javadoc="",
        parent="PointInTime",
        constructible=False,
    ),
    DSLType(
        name="TimeRange",
        javadoc="",
        constructible=False,   # only produced by deserialize_time_range
        constructor_params=[("start", "PointInTime"), ("end", "PointInTime")],
    ),
]}


# ---------------------------------------------------------------------------
# DSL Function Registry
# ---------------------------------------------------------------------------

DSL_FUNCTIONS: dict[str, DSLFunction] = {f.name: f for f in [
    DSLFunction(
        name="deserialize_point_in_time",
        javadoc="",
        param_types=[("serialized_point_in_time", "Literal[str]")],
        return_type="ApproximatePointInTime",
    ),
    DSLFunction(
        name="deserialize_time_range",
        javadoc="",
        param_types=[("serialized_time_range", "Literal[str]")],
        return_type="TimeRange",
    ),
    DSLFunction(
        name="make_precise_point_in_time",
        javadoc="Make a precise point in time from an approximate point in time. Safe to call with an approximate or precise point in time.",
        param_types=[("point_in_time", "PointInTime")],
        return_type="PrecisePointInTime",
    ),
    DSLFunction(
        name="break_avstream_into_two",
        javadoc="Break an audio / video stream into two streams at the given point in time.",
        param_types=[("avstream", "AudioVideoStream"), ("point_in_time", "PointInTime")],
        return_type=["AudioVideoStream", "AudioVideoStream"],
    ),
    DSLFunction(
        name="concatenate_avstreams",
        javadoc="Concatenate two audio / video streams.",
        param_types=[("avstream1", "AudioVideoStream"), ("avstream2", "AudioVideoStream")],
        return_type="AudioVideoStream",
    ),
    DSLFunction(
        name="overlay_avstreams",
        javadoc="Overlay two audio / video streams.",
        param_types=[("avstream1", "AudioVideoStream"), ("avstream2", "AudioVideoStream")],
        return_type="AudioVideoStream",
    ),
    DSLFunction(
        name="emit_output",
        javadoc="",
        param_types=[("final_avstream", "AudioVideoStream")],
        return_type=None,
    ),
]}


# ---------------------------------------------------------------------------
# Literal string validators
#
# Maps:  function_name -> { positional_param_index -> validator_callable }
#
# A validator callable receives the raw string literal (already unquoted) and
# must either return normally (value is valid) or raise ValueError with a
# human-readable message.
# ---------------------------------------------------------------------------

def _validate_point_in_time_str(value: str) -> None:
    """
    Temporary permissive validator: accept any string input.
    """
    _ = value


def _validate_time_range_str(value: str) -> None:
    """
    Temporary permissive validator: accept any string input.
    """
    _ = value


DSL_LITERAL_VALIDATORS: dict[str, dict[int, Callable[[str], None]]] = {
    "deserialize_point_in_time": {0: _validate_point_in_time_str},
    "deserialize_time_range":    {0: _validate_time_range_str},
}
