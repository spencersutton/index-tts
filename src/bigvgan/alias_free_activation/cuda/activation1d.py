# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# load fused CUDA kernel: this enables importing anti_alias_activation_cuda
from typing import cast, override

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx

from bigvgan.alias_free_activation.cuda import load
from bigvgan.alias_free_activation.torch.resample import DownSample1d, UpSample1d

anti_alias_activation_cuda: Any = load.load()


class FusedAntiAliasActivation(torch.autograd.Function):
    """
    Assumes filter size 12, replication padding on upsampling/downsampling, and logscale alpha/beta parameters as inputs.
    The hyperparameters are hard-coded in the kernel to maximize speed.
    NOTE: The fused kenrel is incorrect for Activation1d with different hyperparameters.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx, inputs: Tensor, up_ftr: Tensor, down_ftr: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        return anti_alias_activation_cuda.forward(inputs, up_ftr, down_ftr, alpha, beta)

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        raise NotImplementedError
        return grad_outputs, None, None


class Activation1d(nn.Module):
    up_ratio: int
    down_ratio: int
    act: nn.Module
    upsample: UpSample1d
    downsample: DownSample1d
    fused: bool

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

        self.fused = fused  # Whether to use fused CUDA kernel or not

@override
    def forward(self, x: Tensor) -> Tensor:
        if not self.fused:
            x = self.upsample(x)
            x = self.act(x)
            return self.downsample(x)
        if self.act.__class__.__name__ == "Snake":
            beta = self.act.alpha.data  # Snake uses same params for alpha and beta
        else:
            beta = self.act.beta.data  # Snakebeta uses different params for alpha and beta
        alpha = self.act.alpha.data
        if not self.act.alpha_logscale:  # Exp baked into cuda kernel, cancel it out with a log
            alpha = torch.log(alpha)
            beta = torch.log(beta)

        return FusedAntiAliasActivation.apply(x, self.upsample.filter, self.downsample.lowpass.filter, alpha, beta)
