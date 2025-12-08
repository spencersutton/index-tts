from typing import Literal

from torchcodec import FrameBatch

def clips_at_random_timestamps(
    decoder,
    *,
    num_clips: int = ...,
    num_frames_per_clip: int = ...,
    seconds_between_frames: float | None = ...,
    sampling_range_start: float | None = ...,
    sampling_range_end: float | None = ...,
    policy: Literal["repeat_last", "wrap", "error"] = ...,
) -> FrameBatch: ...
def clips_at_regular_timestamps(
    decoder,
    *,
    seconds_between_clip_starts: float,
    num_frames_per_clip: int = ...,
    seconds_between_frames: float | None = ...,
    sampling_range_start: float | None = ...,
    sampling_range_end: float | None = ...,
    policy: Literal["repeat_last", "wrap", "error"] = ...,
) -> FrameBatch: ...

_COMMON_DOCS = ...
_NUM_CLIPS_DOCS = ...
_SECONDS_BETWEEN_CLIP_STARTS = ...
_NOTE_DOCS = ...
