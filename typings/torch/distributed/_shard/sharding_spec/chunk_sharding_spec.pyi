import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, TypeAlias
from .api import ShardingSpec
from torch.distributed._shard.sharded_tensor import ShardedTensor

if TYPE_CHECKING: ...

@dataclass
class ChunkShardingSpec(ShardingSpec):
    type ShardingDim = int | str
    dim: ShardingDim
    placements: list[torch.distributed._remote_device | str]
    def __post_init__(self):  # -> None:
        ...
    def build_metadata(
        self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties
    ) -> sharded_tensor_meta.ShardedTensorMetadata: ...
    def shard(self, tensor: torch.Tensor, src_rank: int = ..., process_group=...) -> ShardedTensor: ...
