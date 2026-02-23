# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

from typing import override

from torch import Tensor, nn

from bigvgan.activations import Snake
from bigvgan.alias_free_activation.torch.resample import DownSample1d, UpSample1d
from indextts.util import patch_call


class Activation1d(nn.Module):
    act: Snake
    down_ratio: int
    downsample: DownSample1d
    up_ratio: int
    upsample: UpSample1d

    def __init__(
        self,
        activation: Snake,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()

        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
