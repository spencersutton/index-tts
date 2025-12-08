# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

import functools
import typing
from typing import Any

import torch
from numpy import NDArray, float64
from torch import Tensor, nn

class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        n_mels: list[int] = ...,
        window_lengths: list[int] = ...,
        loss_fn: typing.Callable = ...,
        clamp_eps: float = ...,
        mag_weight: float = ...,
        log_weight: float = ...,
        pow: float = ...,
        weight: float = ...,
        match_stride: bool = ...,
        mel_fmin: list[float] = ...,
        mel_fmax: list[float] = ...,
        window_type: str = ...,
    ) -> None: ...
    @staticmethod
    @functools.lru_cache(None)
    def get_window(window_type, window_length) -> NDArray[float64] | Any: ...
    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax) -> NDArray[Any, Any]: ...
    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ) -> Tensor: ...
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

def feature_loss(fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]) -> torch.Tensor: ...
def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]: ...
def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]: ...
