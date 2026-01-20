from typing import Any

from _typeshed import Incomplete
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from indextts.util import patch_call

from .lazy import LazyModuleMixin
from .module import Module

__all__ = ["Bilinear", "Identity", "LazyLinear", "Linear"]

class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class Linear(Module):
    __constants__: Incomplete
    in_features: int
    out_features: int
    weight: Tensor
    bias: Incomplete
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...
    def extra_repr(self) -> str: ...

class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...

class Bilinear(Module):
    __constants__: Incomplete
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor
    bias: Incomplete
    def __init__(
        self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input1: Tensor, input2: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class LazyLinear(LazyModuleMixin, Linear):
    cls_to_become = Linear
    weight: UninitializedParameter
    bias: UninitializedParameter
    out_features: Incomplete
    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    in_features: Incomplete
    def initialize_parameters(self, input) -> None: ...
