import torch
import torch.distributed.tensor._dispatch as op_dispatch
import torch.nn as nn
from collections.abc import Sequence
from typing import Any, Callable, Optional, Self
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement

__all__ = ["DTensor", "distribute_tensor", "distribute_module", "ones", "empty", "full", "rand", "randn", "zeros"]
aten = ...

class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: DTensor, grad_placements: Optional[Sequence[Placement]]):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # -> tuple[DTensor, None]:
        ...

class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        run_check: bool,
        shape: Optional[torch.Size] = ...,
        stride: Optional[tuple[int, ...]] = ...,
    ) -> DTensor: ...
    @staticmethod
    def backward(ctx, grad_output: DTensor):  # -> tuple[Tensor, None, None, None, None, None]:
        ...

class DTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ...
    _op_dispatcher: op_dispatch.OpDispatcher = ...
    @staticmethod
    @torch._disable_dynamo
    def __new__(cls, local_tensor: torch.Tensor, spec: DTensorSpec, *, requires_grad: bool) -> Self: ...
    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def __tensor_flatten__(self):  # -> tuple[list[str], tuple[DTensorSpec, Any]]:

        ...
    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):  # -> DTensor:
        ...
    def __coerce_tangent_metadata__(self):  # -> Self | DTensor:
        ...
    def __coerce_same_metadata_as_tangent__(self, flatten_spec, expected_type=...):  # -> DTensor | None:
        ...
    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=..., kwargs=...):  # -> object:
        ...
    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = ...,
        placements: Optional[Sequence[Placement]] = ...,
        *,
        run_check: bool = ...,
        shape: Optional[torch.Size] = ...,
        stride: Optional[tuple[int, ...]] = ...,
    ) -> DTensor: ...
    def to_local(self, *, grad_placements: Optional[Sequence[Placement]] = ...) -> torch.Tensor: ...
    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = ...,
        placements: Optional[Sequence[Placement]] = ...,
        *,
        async_op: bool = ...,
        forward_dtype: Optional[torch.dtype] = ...,
        backward_dtype: Optional[torch.dtype] = ...,
    ) -> DTensor: ...
    def full_tensor(self, *, grad_placements: Optional[Sequence[Placement]] = ...) -> torch.Tensor: ...
    @property
    def device_mesh(self) -> DeviceMesh: ...
    @property
    def placements(self) -> tuple[Placement, ...]: ...
    def __create_write_items__(self, fqn: str, object: Any):  # -> list[Any]:
        ...
    def __create_chunk_list__(self):  # -> list[Any]:

        ...
    def __get_tensor_shard__(self, index):  # -> Tensor:
        ...

def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
    *,
    src_data_rank: Optional[int] = ...,
) -> DTensor: ...
def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = ...,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = ...,
    input_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = ...,
    output_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = ...,
) -> nn.Module: ...
def ones(
    *size,
    dtype: Optional[torch.dtype] = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
) -> DTensor: ...
def empty(
    *size,
    dtype: Optional[torch.dtype] = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
) -> DTensor: ...
def full(
    size,
    fill_value,
    *,
    dtype: Optional[torch.dtype] = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
) -> DTensor: ...
def rand(
    *size,
    requires_grad: bool = ...,
    dtype: Optional[torch.dtype] = ...,
    layout: torch.layout = ...,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
) -> DTensor: ...
def randn(
    *size,
    requires_grad: bool = ...,
    dtype: Optional[torch.dtype] = ...,
    layout: torch.layout = ...,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
) -> DTensor: ...
def zeros(
    *size,
    requires_grad: bool = ...,
    dtype: Optional[torch.dtype] = ...,
    layout: torch.layout = ...,
    device_mesh: Optional[DeviceMesh] = ...,
    placements: Optional[Sequence[Placement]] = ...,
) -> DTensor: ...
