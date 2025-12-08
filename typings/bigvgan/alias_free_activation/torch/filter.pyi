# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

import torch
from torch import Tensor, nn

if "sinc" in dir(torch):
    sinc = ...
else:
    def sinc(x: torch.Tensor) -> Tensor: ...

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): ...

class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=...,
        half_width=...,
        stride: int = ...,
        padding: bool = ...,
        padding_mode: str = ...,
        kernel_size: int = ...,
    ) -> None: ...
    def forward(self, x) -> Tensor: ...
