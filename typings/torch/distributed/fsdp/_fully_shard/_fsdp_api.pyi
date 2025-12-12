import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union, TypeAlias

_ReduceOp: TypeAlias = Union[dist.ReduceOp, dist.ReduceOp.RedOpType]

@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: Optional[torch.dtype] = ...
    reduce_dtype: Optional[torch.dtype] = ...
    output_dtype: Optional[torch.dtype] = ...
    cast_forward_inputs: bool = ...

class Comm(ABC):
    @abstractmethod
    def allocate(
        self, size: Sequence[Union[int, torch.SymInt]], *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor: ...

class AllGather(Comm):
    @abstractmethod
    def __call__(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, group: dist.ProcessGroup, async_op: bool = ...
    ) -> Optional[dist.Work]: ...

class ReduceScatter(Comm):
    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = ...,
    ) -> Optional[dist.Work]: ...

@dataclass
class OffloadPolicy: ...

@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    pin_memory: bool = ...
