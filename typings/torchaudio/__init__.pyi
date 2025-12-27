import os
from typing import BinaryIO

import torch

from . import (
    compliance,
    datasets,
    functional,
    models,
    pipelines,
    transforms,
    utils,
)
from ._torchcodec import load_with_torchcodec, save_with_torchcodec

def load(
    uri: BinaryIO | str | os.PathLike,
    frame_offset: int = ...,
    num_frames: int = ...,
    normalize: bool = ...,
    channels_first: bool = ...,
    format: str | None = ...,
    buffer_size: int = ...,
    backend: str | None = ...,
) -> tuple[torch.Tensor, int]: ...
def save(
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

__all__ = [
    "compliance",
    "datasets",
    "functional",
    "load",
    "load_with_torchcodec",
    "models",
    "pipelines",
    "save",
    "save_with_torchcodec",
    "transforms",
    "utils",
]
