from typing import Optional, Tuple

import torch
from torchaudio._internal import module_utils as _mod_utils

from .common import AudioMetaData

"""The new soundfile backend which will become default in 0.8.0 onward"""
_IS_SOUNDFILE_AVAILABLE = ...
if _mod_utils.is_module_available("soundfile"):
    _requires_soundfile = ...
    _IS_SOUNDFILE_AVAILABLE = ...
else:
    _requires_soundfile = ...
_SUBTYPE_TO_BITS_PER_SAMPLE = ...
_SUBTYPE_TO_ENCODING = ...

@_requires_soundfile
def info(filepath: str, format: str | None = ...) -> AudioMetaData: ...

_SUBTYPE2DTYPE = ...

@_requires_soundfile
def load(
    filepath: str,
    frame_offset: int = ...,
    num_frames: int = ...,
    normalize: bool = ...,
    channels_first: bool = ...,
    format: str | None = ...,
) -> tuple[torch.Tensor, int]: ...
@_requires_soundfile
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
