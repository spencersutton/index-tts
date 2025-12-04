# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

from typing import Any

import torch
import torch.utils.data
from torch import Tensor

from .env import AttrDict

MAX_WAV_VALUE = ...

def dynamic_range_compression(x, C=..., clip_val=...) -> Any: ...
def dynamic_range_decompression(x, C=...) -> Any: ...
def dynamic_range_compression_torch(x, C=..., clip_val=...) -> Tensor: ...
def dynamic_range_decompression_torch(x, C=...) -> Tensor: ...
def spectral_normalize_torch(magnitudes) -> Tensor: ...
def spectral_de_normalize_torch(magnitudes) -> Tensor: ...

mel_basis_cache = ...
hann_window_cache = ...

def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = ...,
    center: bool = ...,
) -> torch.Tensor: ...
def get_mel_spectrogram(wav, h) -> Tensor: ...
def get_dataset_filelist(a) -> tuple[list[str], list[str], list[Any]]: ...

class MelDataset(torch.utils.data.Dataset[object]):
    def __init__(
        self,
        training_files: list[str],
        hparams: AttrDict,
        segment_size: int,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        sampling_rate: int,
        fmin: int,
        fmax: int | None,
        split: bool = ...,
        shuffle: bool = ...,
        device: str = ...,
        fmax_loss: int | None = ...,
        fine_tuning: bool = ...,
        base_mels_path: str = ...,
        is_seen: bool = ...,
    ) -> None: ...
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]: ...
    def __len__(self) -> int: ...
