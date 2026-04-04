from dataclasses import dataclass

@dataclass
class AudioVideoStream:
  """
  A stream of audio and / or video. (So could be audio only, video only, or audio and video.)
  """
  file_id: str
  namespace: str

@dataclass
class PointInTime:
  pass

@dataclass
class ApproximatePointInTime(PointInTime):
  possible_start_timestamp_milliseconds: int
  possible_end_timestamp_milliseconds: int
  left_is_tight: bool
  right_is_tight: bool

@dataclass
class PrecisePointInTime(PointInTime):
  timestamp_milliseconds: int

@dataclass
class TimeRange:
  start: PointInTime
  end: PointInTime
