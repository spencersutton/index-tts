from collections.abc import Iterator

from torch import Tensor
from torio.io._streaming_media_encoder import CodecConfig

class _StreamingIOBuffer:
    def __init__(self) -> None: ...
    def write(self, b: bytes):  # -> int:
        ...
    def pop(self, n):  # -> Literal[b""]:
        ...

class _AudioStreamingEncoder:
    def __init__(
        self,
        src: Tensor,
        sample_rate: int,
        effect: str,
        muxer: str,
        encoder: str | None,
        codec_config: CodecConfig | None,
        frames_per_chunk: int,
    ) -> None: ...
    def read(self, n):  # -> Literal[b""]:
        ...

class AudioEffector:
    def __init__(
        self,
        effect: str | None = ...,
        format: str | None = ...,
        *,
        encoder: str | None = ...,
        codec_config: CodecConfig | None = ...,
        pad_end: bool = ...,
    ) -> None: ...
    def apply(
        self,
        waveform: Tensor,
        sample_rate: int,
        output_sample_rate: int | None = ...,
    ) -> Tensor: ...
    def stream(
        self,
        waveform: Tensor,
        sample_rate: int,
        frames_per_chunk: int,
        output_sample_rate: int | None = ...,
    ) -> Iterator[Tensor]: ...
