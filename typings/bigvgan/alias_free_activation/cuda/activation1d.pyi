# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

from typing import Any

import torch
from torch import nn

anti_alias_activation_cuda = ...

class FusedAntiAliasActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, up_ftr, down_ftr, alpha, beta): ...
    @staticmethod
    def backward(ctx, output_grads): ...

class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = ...,
        down_ratio: int = ...,
        up_kernel_size: int = ...,
        down_kernel_size: int = ...,
        fused: bool = ...,
    ) -> None: ...
    def forward(self, x) -> Any | None: ...
