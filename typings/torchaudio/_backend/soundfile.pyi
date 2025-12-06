import os
from typing import BinaryIO

import torch
from torchaudio.io import CodecConfig

from .backend import Backend
from .common import AudioMetaData

class SoundfileBackend(Backend):
    @staticmethod
    def info(uri: BinaryIO | str | os.PathLike, format: str | None, buffer_size: int = ...) -> AudioMetaData: ...
    @staticmethod
    def load(
        uri: BinaryIO | str | os.PathLike,
        frame_offset: int = ...,
        num_frames: int = ...,
        normalize: bool = ...,
        channels_first: bool = ...,
        format: str | None = ...,
        buffer_size: int = ...,
    ) -> tuple[torch.Tensor, int]: ...
    @staticmethod
    def save(
        uri: BinaryIO | str | os.PathLike,
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = ...,
        format: str | None = ...,
        encoding: str | None = ...,
        bits_per_sample: int | None = ...,
        buffer_size: int = ...,
        compression: CodecConfig | float | None = ...,
    ) -> None: ...
    @staticmethod
    def can_decode(uri, format) -> bool: ...
    @staticmethod
    def can_encode(uri, format) -> bool: ...
