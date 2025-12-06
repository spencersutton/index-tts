import os
from typing import BinaryIO

import torch
import torchaudio

from .backend import Backend
from .common import AudioMetaData

sox_ext = ...

class SoXBackend(Backend):
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
        compression: torchaudio.io.CodecConfig | float | None = ...,
    ) -> None: ...
    @staticmethod
    def can_decode(uri: BinaryIO | str | os.PathLike, format: str | None) -> bool: ...
    @staticmethod
    def can_encode(uri: BinaryIO | str | os.PathLike, format: str | None) -> bool: ...
