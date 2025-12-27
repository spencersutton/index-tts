import contextlib
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.distributed.tensor import Shard
from torch.utils.hooks import RemovableHandle

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_collectives import AllGatherResult
from ._fsdp_common import FSDPMeshInfo, TrainingState

logger = ...
type _ModuleToHandleDict = dict[nn.Module, RemovableHandle]

class FSDPCommContext:
    """This has the communication state shared across FSDP states/parameter groups."""
    def lazy_init(self, device: torch.device): ...
    def get_all_gather_streams(
        self, async_op: bool, training_state: TrainingState
    ) -> tuple[torch.Stream, torch.Stream]: ...

class AllGatherState(NamedTuple):
    """AllGatherState(all_gather_result, event)"""

    all_gather_result: AllGatherResult
    event: torch.Event | None

class ReduceScatterState(NamedTuple):
    """ReduceScatterState(reduce_scatter_input, event)"""

    reduce_scatter_input: torch.Tensor
    event: torch.Event | None

class AllReduceState(NamedTuple):
    """AllReduceState(all_reduce_input, event)"""

    all_reduce_input: torch.Tensor
    event: torch.Event | None

class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype | None
    _reduce_dtype: torch.dtype | None
    def __init__(
        self,
        params: list[nn.Parameter],
        modules: tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ) -> None: ...
    def lazy_init(self): ...
    def set_allocate_memory_from_process_group(self, enable: bool) -> None:
        """
        Whether to (try to) use the ProcessGroup's allocate_tensor method for
        the staging buffers for collective comms.
        """
    def unshard(self, async_op: bool = ...): ...
    def wait_for_unshard(self):
        """
        1. In forward with implicit prefetching, to overlap the current copy-out
        with the next all-gather, we save a reference to the current all-gather
        result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
        all-gather result immediately after the current copy-out since we can
        already overlap the current copy-out with the previous reduce-scatter.
        """
    def reshard(self): ...
    def pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def post_forward(self, module: nn.Module, input: Any, output: Any): ...
    def pre_backward(self, default_prefetch: bool, *unused: Any): ...
    def post_backward(self, *unused: Any): ...
    def finalize_backward(self): ...
    @property
    def is_sharded(self) -> bool: ...
    @property
    def is_sharded_post_forward(self) -> bool: ...
    @property
    def is_unsharded(self) -> bool: ...
    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState): ...

class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor): ...
    @staticmethod
    def backward(ctx, *grads: torch.Tensor): ...
