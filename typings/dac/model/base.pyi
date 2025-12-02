from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from audiotools import AudioSignal

SUPPORTED_VERSIONS = ...

@dataclass
class DACFile:
    codes: torch.Tensor
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str
    def save(self, path):  # -> Path:
        ...
    @classmethod
    def load(cls, path):  # -> Self:
        ...

class CodecMixin:
    @property
    def padding(self):  # -> bool:
        ...
    @padding.setter
    def padding(self, value):  # -> None:
        ...
    def get_delay(self):  # -> int:
        ...
    def get_output_length(self, input_length): ...
    @torch.no_grad()
    def compress(
        self,
        audio_path_or_signal: str | Path | AudioSignal,
        win_duration: float = ...,
        verbose: bool = ...,
        normalize_db: float = ...,
        n_quantizers: int = ...,
    ) -> DACFile: ...
    @torch.no_grad()
    def decompress(self, obj: str | Path | DACFile, verbose: bool = ...) -> AudioSignal: ...
