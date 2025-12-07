from pathlib import Path
from typing import Any

from torch import Tensor

class VideoEncoder:
    def __init__(self, frames: Tensor, *, frame_rate: float) -> None: ...
    def to_file(
        self,
        dest: str | Path,
        *,
        codec: str | None = ...,
        pixel_format: str | None = ...,
        crf: float | None = ...,
        preset: str | int | None = ...,
        extra_options: dict[str, Any] | None = ...,
    ) -> None: ...
    def to_tensor(
        self,
        format: str,
        *,
        codec: str | None = ...,
        pixel_format: str | None = ...,
        crf: float | None = ...,
        preset: str | int | None = ...,
        extra_options: dict[str, Any] | None = ...,
    ) -> Tensor: ...
    def to_file_like(
        self,
        file_like,
        format: str,
        *,
        codec: str | None = ...,
        pixel_format: str | None = ...,
        crf: float | None = ...,
        preset: str | int | None = ...,
        extra_options: dict[str, Any] | None = ...,
    ) -> None: ...
