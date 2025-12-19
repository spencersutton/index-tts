# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.
from typing import Any, override

import torch
from torch import Tensor, nn

from indextts.util import patch_call

from ...activations import Snake, SnakeBeta
from ..torch.resample import DownSample1d, UpSample1d
from . import load

# load fused CUDA kernel: this enables importing anti_alias_activation_cuda
anti_alias_activation_cuda = load.load()


class FusedAntiAliasActivation(torch.autograd.Function):
    """Assumes filter size 12, replication padding on upsampling/downsampling, and logscale alpha/beta parameters as inputs.
    The hyperparameters are hard-coded in the kernel to maximize speed.
    NOTE: The fused kenrel is incorrect for Activation1d with different hyperparameters.
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, up_ftr: Tensor, down_ftr: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:  # noqa: ARG004
        return anti_alias_activation_cuda.forward(inputs, up_ftr, down_ftr, alpha, beta)

    @staticmethod
    def backward(ctx: Any, *output_grads: Any) -> Any:  # noqa: ARG004
        raise NotImplementedError
        return output_grads, None, None


class Activation1d(nn.Module):
    up_ratio: int
    down_ratio: int
    act: Snake | SnakeBeta
    upsample: UpSample1d
    downsample: DownSample1d
    fused: bool

    def __init__(
        self,
        activation: Snake | SnakeBeta,
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
        if isinstance(self.act, Snake):
            beta = self.act.alpha.data  # Snake uses same params for alpha and beta
        else:
            beta = self.act.beta.data  # Snakebeta uses different params for alpha and beta
        alpha = self.act.alpha.data
        if not self.act.alpha_logscale:  # Exp baked into cuda kernel, cancel it out with a log
            alpha = torch.log(alpha)
            beta = torch.log(beta)

        return FusedAntiAliasActivation.apply(x, self.upsample.filter, self.downsample.lowpass.filter, alpha, beta)

    @patch_call(forward)
    def __call__(self) -> None: ...
