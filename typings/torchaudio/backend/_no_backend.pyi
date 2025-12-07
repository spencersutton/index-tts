from collections.abc import Callable
from pathlib import Path

from torch import Tensor
from torchaudio import AudioMetaData

def load(
    filepath: str | Path,
    out: Tensor | None = ...,
    normalization: bool | float | Callable = ...,
    channels_first: bool = ...,
    num_frames: int = ...,
    offset: int = ...,
    filetype: str | None = ...,
) -> tuple[Tensor, int]: ...
def save(
    filepath: str,
    src: Tensor,
    sample_rate: int,
    precision: int = ...,
    channels_first: bool = ...,
) -> None: ...
def info(filepath: str) -> AudioMetaData: ...
