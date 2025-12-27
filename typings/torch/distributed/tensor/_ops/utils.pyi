from collections.abc import Callable, Iterable, Sequence
from typing import ParamSpec, TypeVar

import torch
from torch._prims_common import DimsSequenceType, DimsType
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

_T = TypeVar("_T")
_P = ParamSpec("_P")

def register_prop_rule(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload], schema_info: RuntimeSchemaInfo | None = ...
) -> Callable[[Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]]: ...
def register_op_strategy(op, schema_info=...) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def replicate_op_strategy(op_schema: OpSchema) -> StrategyType:
    """Fallback strategy all use Replication()"""

def as_list(x: list[object] | object) -> list[object] | torch.fx.immutable_collections.immutable_list: ...
def normalize_dim(dim: int, ndim: int) -> int: ...
def normalize_dims(dims: DimsType, ndim: int) -> DimsSequenceType:
    """Normalize a dim or a sequence of dims, so that they are all positive."""

def prod(xs: Iterable[int]) -> int: ...
def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is shardable according to the spec."""

def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is evenly shardable according to the spec."""

def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded."""

def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh."""

def infer_broadcast_dims_map(common_shape: torch.Size, input_shape: torch.Size) -> list[int]: ...
def map_placements_after_broadcast(
    placements: tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: list[int],
    partial_to_replicate: bool = ...,
) -> tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""

def generate_redistribute_costs(src_strategy: OpStrategy, dst_spec: DTensorSpec) -> list[float]:
    """
    Generates one row in the 'redistribute_costs' matrix in an OpSpec
    The length of the returned list will match the number of strategies in 'src_strategy'.

    Each value in the row is the cost of redistributing from a particular src_strategy to dst_spec.
    """

def expand_to_full_mesh_op_strategy(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_mesh_dim_strategies: list[PlacementList],
    *,
    input_index: int = ...,
    inplace_op: bool = ...,
    is_valid_strategy_cb: Callable[[list[DTensorSpec], tuple[DTensorSpec | None, ...]], bool] | None = ...,
) -> OpStrategy:
    """
    Convenience function to allow writing a sharding strategy considering only a single mesh dimension,
    and have it expanded combinatorically to all mesh dimensions.

    Args:
        mesh (DeviceMesh): the device mesh to expand the strategy to
        op_schema (OpSchema): the op schema
        single_mesh_dim_strategies (list[PlacementList]): the sharding strategies to expand. The outer list is over
            different strategies.  The inner PlacementList is over the outputs and inputs of the op. If input_index is 1,
            a PlacementList looks like [output_placement, input_placement1, input_placement2, ...].
        input_index: the number of outputs of the op, defaults to 1
        inplace_op: whether the op is inplace or not, defaults to False
        is_valid_strategy_cb: a callback function to filter out invalid sharding rules, defaults to None.

    Example: Let's say `my_op(tensor_x, tensor_y) - > output_tensor`  can support sharding or replicating tensor_x,
    but always requires tensor_y to be replicated.  We can specify these valid combinations ignoring mesh dims.
    Then, we can rely on `expand_to_full_mesh_op_strategy` to create every possible combination of these shardings
    over multiple mesh dimensions, filtering out any combinations that are invalid based on the actual mesh dim size.

        single_mesh_dim_strategies = [
            # first strategy: return output sharded on first dim, shard tensor_x on its first dim, replicate tensor_y
            [Shard(0), Shard(0), Replicate()]
            # second strategy: replicate output, and both inputs
            [Replicate(), Replicate(), Replicate()]
        ]
    """
