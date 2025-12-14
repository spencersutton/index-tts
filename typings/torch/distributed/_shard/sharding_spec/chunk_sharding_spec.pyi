from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, Union

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.sharded_tensor import ShardedTensor

from .api import ShardingSpec

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
