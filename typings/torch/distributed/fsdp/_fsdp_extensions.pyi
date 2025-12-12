import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Any, Optional
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.tensor import DTensor, DeviceMesh

class FSDPExtensions(ABC):
    @abstractmethod
    def pre_flatten_transform(self, tensor: torch.Tensor) -> tuple[torch.Tensor, Optional[Any]]: ...
    @abstractmethod
    def post_unflatten_transform(self, tensor: torch.Tensor, param_extension: Any) -> torch.Tensor: ...
    @abstractmethod
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = ...,
    ) -> torch.Tensor: ...
    @abstractmethod
    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor: ...
    @abstractmethod
    def pre_load_state_dict_transform(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list[Shard]]: ...
    @abstractmethod
    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: Optional[DeviceMesh]) -> torch.Tensor: ...

_extensions: Optional[FSDPExtensions] = ...
