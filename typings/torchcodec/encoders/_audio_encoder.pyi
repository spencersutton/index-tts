from pathlib import Path

from torch import Tensor

class AudioEncoder:
    def __init__(self, samples: Tensor, *, sample_rate: int) -> None: ...
    def to_file(
        self,
        dest: str | Path,
        *,
        bit_rate: int | None = ...,
        num_channels: int | None = ...,
        sample_rate: int | None = ...,
    ) -> None: ...
    def to_tensor(
        self,
        format: str,
        *,
        bit_rate: int | None = ...,
        num_channels: int | None = ...,
        sample_rate: int | None = ...,
    ) -> Tensor: ...
    def to_file_like(
        self,
        file_like,
        format: str,
        *,
        bit_rate: int | None = ...,
        num_channels: int | None = ...,
        sample_rate: int | None = ...,
    ) -> None: ...
