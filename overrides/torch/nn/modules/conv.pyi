from _typeshed import Incomplete
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import UninitializedParameter

from indextts.util import patch_call

from .lazy import LazyModuleMixin
from .module import Module

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
]

class _ConvNd(Module):
    __constants__: Incomplete
    __annotations__: Incomplete
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Tensor | None
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self): ...

class Conv1d(_ConvNd):
    __doc__: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class Conv2d(_ConvNd):
    __doc__: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class Conv3d(_ConvNd):
    __doc__: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class _ConvTransposeNd(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ) -> None: ...

class ConvTranspose1d(_ConvTransposeNd):
    __doc__: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class ConvTranspose2d(_ConvTransposeNd):
    __doc__: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class ConvTranspose3d(_ConvTransposeNd):
    __doc__: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class _ConvTransposeMixin(_ConvTransposeNd):
    def __init__(self, *args, **kwargs) -> None: ...

class _LazyConvXdMixin(LazyModuleMixin):
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input: Tensor, *args, **kwargs) -> None: ...

class LazyConv1d(_LazyConvXdMixin, Conv1d):
    cls_to_become = Conv1d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...

class LazyConv2d(_LazyConvXdMixin, Conv2d):
    cls_to_become = Conv2d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...

class LazyConv3d(_LazyConvXdMixin, Conv3d):
    cls_to_become = Conv3d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...

class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):
    cls_to_become = ConvTranspose1d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...

class LazyConvTranspose2d(_LazyConvXdMixin, ConvTranspose2d):
    cls_to_become = ConvTranspose2d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...

class LazyConvTranspose3d(_LazyConvXdMixin, ConvTranspose3d):
    cls_to_become = ConvTranspose3d
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
