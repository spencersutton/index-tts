from typing import Literal

from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder

def clips_at_random_indices(
    decoder: VideoDecoder,
    *,
    num_clips: int = ...,
    num_frames_per_clip: int = ...,
    num_indices_between_frames: int = ...,
    sampling_range_start: int = ...,
    sampling_range_end: int | None = ...,
    policy: Literal["repeat_last", "wrap", "error"] = ...,
) -> FrameBatch: ...
def clips_at_regular_indices(
    decoder: VideoDecoder,
    *,
    num_clips: int = ...,
    num_frames_per_clip: int = ...,
    num_indices_between_frames: int = ...,
    sampling_range_start: int = ...,
    sampling_range_end: int | None = ...,
    policy: Literal["repeat_last", "wrap", "error"] = ...,
) -> FrameBatch: ...

_COMMON_DOCS = ...
