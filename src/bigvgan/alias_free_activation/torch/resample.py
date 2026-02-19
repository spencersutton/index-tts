# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

from typing import override

from torch import Tensor, nn
from torch.nn import functional as F

from bigvgan.alias_free_activation.torch.filter import LowPassFilter1d, kaiser_sinc_filter1d
from indextts.util import patch_call


class UpSample1d(nn.Module):
    filter: Tensor
    kernel_size: int
    pad_left: int
    pad_right: int
    pad: int
    ratio: int
    stride: int

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()

        self.ratio = ratio
        self.kernel_size = kernel_size or (6 * ratio // 2) * 2
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left, self.pad_right = (self.pad * ratio + (self.kernel_size - ratio + k) // 2 for k in (0, 1))
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.filter = nn.Buffer(filter)

    # x: [B, C, T]
    @override
    def forward(self, x: Tensor) -> Tensor:
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return x[..., self.pad_left : -self.pad_right]

    @patch_call(forward)
    def __call__(self) -> None: ...


class DownSample1d(nn.Module):
    ratio: int
    kernel_size: int
    lowpass: LowPassFilter1d

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.lowpass.__call__(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
