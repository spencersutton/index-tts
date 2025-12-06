
import torch
from torchaudio import AudioMetaData

sox_ext = ...

def info(filepath: str, format: str | None = ...) -> AudioMetaData: ...
def load(
    filepath: str,
    frame_offset: int = ...,
    num_frames: int = ...,
    normalize: bool = ...,
    channels_first: bool = ...,
    format: str | None = ...,
) -> tuple[torch.Tensor, int]: ...
def save(
    filepath: str,
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = ...,
    compression: float | None = ...,
    format: str | None = ...,
    encoding: str | None = ...,
    bits_per_sample: int | None = ...,
):  # -> None:
    ...
