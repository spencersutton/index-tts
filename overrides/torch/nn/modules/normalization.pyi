import torch
from _typeshed import Incomplete
from torch import Tensor

from indextts.util import patch_call

from .module import Module

__all__ = ["CrossMapLRN2d", "GroupNorm", "LayerNorm", "LocalResponseNorm", "RMSNorm"]

class LocalResponseNorm(Module):
    __constants__: Incomplete
    size: int
    alpha: float
    beta: float
    k: float
    def __init__(self, size: int, alpha: float = 0.0001, beta: float = 0.75, k: float = 1.0) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float
    def __init__(self, size: int, alpha: float = 0.0001, beta: float = 0.75, k: float = 1) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class LayerNorm(Module):
    __constants__: Incomplete
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool
    weight: Incomplete
    bias: Incomplete
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class GroupNorm(Module):
    __constants__: Incomplete
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    weight: Incomplete
    bias: Incomplete
    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True, device=None, dtype=None
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class RMSNorm(Module):
    __constants__: Incomplete
    normalized_shape: tuple[int, ...]
    eps: float | None
    elementwise_affine: bool
    weight: Incomplete
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...
    @patch_call(forward)
    def __call__(self) -> None: ...
