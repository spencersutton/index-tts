import os
from abc import ABC, abstractmethod
from typing import BinaryIO

from torch import Tensor
from torchaudio.io import CodecConfig

from .common import AudioMetaData

class Backend(ABC):
    @staticmethod
    @abstractmethod
    def info(
        uri: BinaryIO | str | os.PathLike,
        format: str | None,
        buffer_size: int = ...,
    ) -> AudioMetaData: ...
    @staticmethod
    @abstractmethod
    def load(
        uri: BinaryIO | str | os.PathLike,
        frame_offset: int = ...,
        num_frames: int = ...,
        normalize: bool = ...,
        channels_first: bool = ...,
        format: str | None = ...,
        buffer_size: int = ...,
    ) -> tuple[Tensor, int]: ...
    @staticmethod
    @abstractmethod
    def save(
        uri: BinaryIO | str | os.PathLike,
        src: Tensor,
        sample_rate: int,
        channels_first: bool = ...,
        format: str | None = ...,
        encoding: str | None = ...,
        bits_per_sample: int | None = ...,
        buffer_size: int = ...,
        compression: CodecConfig | float | None = ...,
    ) -> None: ...
    @staticmethod
    @abstractmethod
    def can_decode(uri: BinaryIO | str | os.PathLike, format: str | None) -> bool: ...
    @staticmethod
    @abstractmethod
    def can_encode(uri: BinaryIO | str | os.PathLike, format: str | None) -> bool: ...
