import torch.ao.nn.intrinsic
import torch.ao.nn.quantized as nnq

__all__ = ["BNReLU2d", "BNReLU3d"]

class BNReLU2d(nnq.BatchNorm2d):
    """
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d
    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d
    def __init__(self, num_features, eps=..., momentum=..., device=..., dtype=...) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point) -> Self: ...

class BNReLU3d(nnq.BatchNorm3d):
    """
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d
    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d
    def __init__(self, num_features, eps=..., momentum=..., device=..., dtype=...) -> None: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point) -> Self: ...
