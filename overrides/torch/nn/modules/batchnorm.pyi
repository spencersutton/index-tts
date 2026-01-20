from typing import Any

from _typeshed import Incomplete
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from indextts.util import patch_call

from .lazy import LazyModuleMixin
from .module import Module

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "SyncBatchNorm",
]

class _NormBase(Module):
    __constants__: Incomplete
    num_features: int
    eps: float
    momentum: float | None
    affine: bool
    track_running_stats: bool
    weight: Incomplete
    bias: Incomplete
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_running_stats(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self): ...

class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class _LazyNormBase(LazyModuleMixin, _NormBase):
    weight: UninitializedParameter
    bias: UninitializedParameter
    affine: Incomplete
    track_running_stats: Incomplete
    running_mean: Incomplete
    running_var: Incomplete
    num_batches_tracked: Incomplete
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    num_features: Incomplete
    def initialize_parameters(self, input) -> None: ...

class BatchNorm1d(_BatchNorm): ...

class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    cls_to_become = BatchNorm1d

class BatchNorm2d(_BatchNorm): ...

class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    cls_to_become = BatchNorm2d

class BatchNorm3d(_BatchNorm): ...

class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    cls_to_become = BatchNorm3d

class SyncBatchNorm(_BatchNorm):
    process_group: Incomplete
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Any | None = None,
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...
    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None): ...
