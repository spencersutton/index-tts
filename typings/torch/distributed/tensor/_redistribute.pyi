from typing import NamedTuple

import torch
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

logger = ...

class _TransformInfo(NamedTuple):
    """_TransformInfo(mesh_dim, src_dst_placements, logical_shape)"""

    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    logical_shape: list[int]

def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = ...,
    is_backward: bool = ...,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: dtensor.DTensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool = ...,
        forward_dtype: torch.dtype | None = ...,
        backward_dtype: torch.dtype | None = ...,
    ): ...
    @staticmethod
    def backward(ctx, grad_output: dtensor.DTensor): ...
