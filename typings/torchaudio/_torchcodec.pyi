import os
from typing import BinaryIO, Optional, Tuple, Union

import torch

"""TorchCodec integration for TorchAudio."""

def load_with_torchcodec(
    uri: BinaryIO | str | os.PathLike,
    frame_offset: int = ...,
    num_frames: int = ...,
    normalize: bool = ...,
    channels_first: bool = ...,
    format: str | None = ...,
    buffer_size: int = ...,
    backend: str | None = ...,
) -> tuple[torch.Tensor, int]: ...
def save_with_torchcodec(
    uri: str | os.PathLike,
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = ...,
    format: str | None = ...,
    encoding: str | None = ...,
    bits_per_sample: int | None = ...,
    buffer_size: int = ...,
    backend: str | None = ...,
    compression: float | None = ...,
) -> None: ...
