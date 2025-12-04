from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["MelResNet", "ResBlock", "Stretch2d", "UpsampleNetwork", "WaveRNN"]

class ResBlock(nn.Module):
    def __init__(self, n_freq: int = ...) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class MelResNet(nn.Module):
    def __init__(
        self,
        n_res_block: int = ...,
        n_freq: int = ...,
        n_hidden: int = ...,
        n_output: int = ...,
        kernel_size: int = ...,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class Stretch2d(nn.Module):
    def __init__(self, time_scale: int, freq_scale: int) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class UpsampleNetwork(nn.Module):
    def __init__(
        self,
        upsample_scales: list[int],
        n_res_block: int = ...,
        n_freq: int = ...,
        n_hidden: int = ...,
        n_output: int = ...,
        kernel_size: int = ...,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> tuple[Tensor, Tensor]: ...

class WaveRNN(nn.Module):
    def __init__(
        self,
        upsample_scales: list[int],
        n_classes: int,
        hop_length: int,
        n_res_block: int = ...,
        n_rnn: int = ...,
        n_fc: int = ...,
        kernel_size: int = ...,
        n_freq: int = ...,
        n_hidden: int = ...,
        n_output: int = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor, specgram: Tensor) -> Tensor: ...
    @torch.jit.export
    def infer(self, specgram: Tensor, lengths: Tensor | None = ...) -> tuple[Tensor, Tensor | None]: ...
