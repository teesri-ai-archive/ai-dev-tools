"""
dsl_builtins.py
===============
Real Python implementations of every DSL type and function.

To add a new DSL symbol:
  1. Add the class/function here.
  2. Add it to BUILTIN_IMPLEMENTATIONS at the bottom.
  3. Update dsl_schema.py (DSL_TYPES / DSL_FUNCTIONS).
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class AudioVideoStream:
    file_id: str

    def __repr__(self) -> str:
        return f"AudioVideoStream(file_id={self.file_id!r})"


class PointInTime:
    pass


@dataclass
class ApproximatePointInTime(PointInTime):
    raw: str  # the original serialized representation

    def __repr__(self) -> str:
        return f"ApproximatePointInTime({self.raw!r})"


@dataclass
class PrecisePointInTime(PointInTime):
    raw: str

    def __repr__(self) -> str:
        return f"PrecisePointInTime({self.raw!r})"


@dataclass
class TimeRange:
    start: PointInTime
    end: PointInTime

    def __repr__(self) -> str:
        return f"TimeRange(start={self.start!r}, end={self.end!r})"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def deserialize_point_in_time(serialized_point_in_time: str) -> ApproximatePointInTime:
    """Parse a time string into an ApproximatePointInTime."""
    return ApproximatePointInTime(raw=serialized_point_in_time)


def deserialize_time_range(serialized_time_range: str) -> TimeRange:
    """Parse a time-range string '<start>--<end>' into a TimeRange."""
    start_str, end_str = [s.strip() for s in serialized_time_range.split("--", 1)]
    return TimeRange(
        start=deserialize_point_in_time(start_str),
        end=deserialize_point_in_time(end_str),
    )


def make_precise_point_in_time(
    approximate_time: ApproximatePointInTime,
) -> PrecisePointInTime:
    """Promote an approximate point in time to a precise one (stub)."""
    # In a real implementation this would snap to a frame boundary, etc.
    return PrecisePointInTime(raw=approximate_time.raw)


def break_avstream_into_two(
    avstream: AudioVideoStream,
    point_in_time: PointInTime,
) -> tuple[AudioVideoStream, AudioVideoStream]:
    """Split a stream at point_in_time (stub implementation)."""
    raw = getattr(point_in_time, "raw", str(point_in_time))
    part1 = AudioVideoStream(file_id=f"{avstream.file_id}[0:{raw}]")
    part2 = AudioVideoStream(file_id=f"{avstream.file_id}[{raw}:]")
    return part1, part2


def concatenate_avstreams(
    avstream1: AudioVideoStream,
    avstream2: AudioVideoStream,
) -> AudioVideoStream:
    """Concatenate two streams (stub)."""
    return AudioVideoStream(file_id=f"concat({avstream1.file_id},{avstream2.file_id})")


def overlay_avstreams(
    avstream1: AudioVideoStream,
    avstream2: AudioVideoStream,
) -> AudioVideoStream:
    """Overlay two streams (stub)."""
    return AudioVideoStream(file_id=f"overlay({avstream1.file_id},{avstream2.file_id})")


def emit_output(final_avstream: AudioVideoStream) -> None:
    """Emit the final output (stub – prints to stdout)."""
    print(f"[emit_output] Final stream: {final_avstream!r}")


def dsl_get_attr(obj: object, attr_name: str) -> object:
    """Internal helper used by the parser for validated attribute access."""
    return getattr(obj, attr_name)


def dsl_identity(value: object) -> object:
    """Internal helper used by the parser for constant assignments."""
    return value


# ---------------------------------------------------------------------------
# Implementation registry
# ---------------------------------------------------------------------------
# Maps every DSL-visible name to its Python callable / class.
# The executor looks up callees here.

BUILTIN_IMPLEMENTATIONS: dict[str, object] = {
    # Types (constructible ones)
    "AudioVideoStream": AudioVideoStream,
    # Functions
    "deserialize_point_in_time": deserialize_point_in_time,
    "deserialize_time_range": deserialize_time_range,
    "make_precise_point_in_time": make_precise_point_in_time,
    "break_avstream_into_two": break_avstream_into_two,
    "concatenate_avstreams": concatenate_avstreams,
    "overlay_avstreams": overlay_avstreams,
    "emit_output": emit_output,
    "__dsl_get_attr__": dsl_get_attr,
    "__dsl_identity__": dsl_identity,
}
