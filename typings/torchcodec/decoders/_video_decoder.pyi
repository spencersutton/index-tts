import io
import numbers
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch import device as torch_device
from torchcodec import Frame, FrameBatch

class VideoDecoder:
    def __init__(
        self,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
        *,
        stream_index: int | None = ...,
        dimension_order: Literal["NCHW", "NHWC"] = ...,
        num_ffmpeg_threads: int = ...,
        device: str | torch_device | None = ...,
        seek_mode: Literal["exact", "approximate"] = ...,
        custom_frame_mappings: str | bytes | io.RawIOBase | io.BufferedReader | None = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: numbers.Integral | slice) -> Tensor: ...
    def get_frame_at(self, index: int) -> Frame: ...
    def get_frames_at(self, indices: torch.Tensor | list[int]) -> FrameBatch: ...
    def get_frames_in_range(self, start: int, stop: int, step: int = ...) -> FrameBatch: ...
    def get_frame_played_at(self, seconds: float) -> Frame: ...
    def get_frames_played_at(self, seconds: torch.Tensor | list[float]) -> FrameBatch: ...
    def get_frames_played_in_range(self, start_seconds: float, stop_seconds: float) -> FrameBatch: ...
