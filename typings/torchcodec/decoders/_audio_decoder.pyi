import io
from pathlib import Path

from torch import Tensor

from .._frame import AudioSamples

class AudioDecoder:
    def __init__(
        self,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
        *,
        stream_index: int | None = ...,
        sample_rate: int | None = ...,
        num_channels: int | None = ...,
    ) -> None: ...
    def get_all_samples(self) -> AudioSamples: ...
    def get_samples_played_in_range(
        self, start_seconds: float = ..., stop_seconds: float | None = ...
    ) -> AudioSamples: ...
