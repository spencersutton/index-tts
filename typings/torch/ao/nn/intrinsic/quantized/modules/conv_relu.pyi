import torch.ao.nn.quantized as nnq

__all__ = ["ConvReLU1d", "ConvReLU2d", "ConvReLU3d"]
_reverse_repeat_padding = ...

class ConvReLU1d(nnq.Conv1d):
    """
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d
    """

    _FLOAT_MODULE = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point) -> Self: ...

class ConvReLU2d(nnq.Conv2d):
    """
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d
    """

    _FLOAT_MODULE = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point) -> Self: ...

class ConvReLU3d(nnq.Conv3d):
    """
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv3d`.

    Attributes: Same as torch.ao.nn.quantized.Conv3d
    """

    _FLOAT_MODULE = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point) -> Self: ...
