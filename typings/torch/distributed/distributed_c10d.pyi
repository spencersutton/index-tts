import pickle
import torch
from datetime import timedelta
from typing import Any, Callable, Optional, Union
from typing_extensions import deprecated
from torch._C._distributed_c10d import ProcessGroup, ReduceOp, Store, Work
from .c10d_logger import _exception_logger, _time_logger

"""Distributed Collective Communication (c10d)."""
__all__ = [
    "Backend",
    "BackendConfig",
    "GroupMember",
    "P2POp",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_object",
    "all_reduce",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "send_object_list",
    "recv_object_list",
    "broadcast_object_list",
    "destroy_process_group",
    "gather",
    "gather_object",
    "get_backend_config",
    "get_backend",
    "get_default_backend_for_device",
    "get_rank",
    "get_world_size",
    "get_pg_count",
    "group",
    "init_process_group",
    "irecv",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_backend_available",
    "is_nccl_available",
    "is_torchelastic_launched",
    "is_ucc_available",
    "is_xccl_available",
    "isend",
    "monitored_barrier",
    "new_group",
    "new_subgroups",
    "new_subgroups_by_enumeration",
    "recv",
    "reduce",
    "reduce_scatter",
    "scatter",
    "scatter_object_list",
    "send",
    "supports_complex",
    "AllreduceCoalescedOptions",
    "AllreduceOptions",
    "AllToAllOptions",
    "BarrierOptions",
    "BroadcastOptions",
    "GatherOptions",
    "PrefixStore",
    "ProcessGroup",
    "ReduceOp",
    "ReduceOptions",
    "ReduceScatterOptions",
    "ScatterOptions",
    "Store",
    "DebugLevel",
    "get_debug_level",
    "Work",
    "default_pg_timeout",
    "get_group_rank",
    "get_global_rank",
    "get_process_group_ranks",
    "reduce_op",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "get_node_local_rank",
    "split_group",
]
_MPI_AVAILABLE = ...
_NCCL_AVAILABLE = ...
_GLOO_AVAILABLE = ...
_UCC_AVAILABLE = ...
_XCCL_AVAILABLE = ...
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler
__all__ += ["ProcessGroupMPI"]
__all__ += ["ProcessGroupNCCL"]
__all__ += ["ProcessGroupGloo"]
__all__ += ["ProcessGroupUCC"]
__all__ += ["ProcessGroupXCCL"]
logger = ...
PG_WRAPPER_STORE_PREFIX = ...

def supports_complex(reduceOp: ReduceOp) -> bool: ...

class Backend(str):
    UNDEFINED = ...
    GLOO = ...
    NCCL = ...
    UCC = ...
    MPI = ...
    XCCL = ...
    _BackendPlugin = ...
    _plugins: dict[str, _BackendPlugin] = ...
    backend_list = ...
    default_device_backend_map: dict[str, str] = ...
    backend_capability: dict[str, list[str]] = ...
    backend_type_map: dict[str, ProcessGroup.BackendType] = ...
    def __new__(cls, name: str):  # -> str | Any:

        ...
    @classmethod
    def register_backend(cls, name, func, extended_api=..., devices: Optional[Union[str, list[str]]] = ...) -> None: ...

class BackendConfig:
    def __init__(self, backend: Backend) -> None: ...
    def __repr__(self):  # -> str:

        ...
    def get_device_backend_map(self) -> dict[str, Backend]: ...

class _reduce_op:
    def __init__(self) -> None: ...
    @deprecated(
        "`torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead",
        category=FutureWarning,
    )
    def __getattribute__(self, key):  # -> Any:
        ...

reduce_op = ...

class P2POp:
    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        peer: Optional[int] = ...,
        group: Optional[ProcessGroup] = ...,
        tag: int = ...,
        group_peer: Optional[int] = ...,
    ) -> None: ...
    def __new__(
        cls,
        op: Callable,
        tensor: torch.Tensor,
        peer: Optional[int] = ...,
        group: Optional[ProcessGroup] = ...,
        tag: int = ...,
        group_peer: Optional[int] = ...,
    ):  # -> Self:

        ...
    def __repr__(self):  # -> str:
        ...

class _CollOp:
    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        dst_tensor: Optional[torch.Tensor] = ...,
        redop: Optional[ReduceOp] = ...,
        root: Optional[int] = ...,
    ) -> None: ...

_pg_map: dict[ProcessGroup, tuple[str, Store]] = ...
_pg_names: dict[ProcessGroup, str] = ...
_pg_group_ranks: dict[ProcessGroup, dict[int, int]] = ...
_pg_backend_config: dict[ProcessGroup, str] = ...
_group_count = ...
_tags_to_pg: dict[str, list[ProcessGroup]] = ...
_pg_to_tag: dict[ProcessGroup, str] = ...
_backend: Optional[str] = ...

class _World:
    def __init__(self) -> None: ...
    @property
    def default_pg(self) -> Optional[ProcessGroup]: ...
    @default_pg.setter
    def default_pg(self, value) -> None: ...
    @property
    def pg_map(self) -> dict[ProcessGroup, tuple[str, Store]]: ...
    @property
    def pg_names(self) -> dict[ProcessGroup, str]: ...
    @property
    def pg_group_ranks(self) -> dict[ProcessGroup, dict[int, int]]: ...
    @property
    def pg_backend_config(self) -> dict[ProcessGroup, str]: ...
    @property
    def group_count(self) -> int: ...
    @group_count.setter
    def group_count(self, value: int) -> None: ...
    @property
    def tags_to_pg(self) -> dict[str, list[ProcessGroup]]: ...
    @property
    def pg_to_tag(self) -> dict[ProcessGroup, str]: ...
    @property
    def pg_coalesce_state(self) -> dict[ProcessGroup, list[_CollOp]]: ...
    @property
    def pg_config_info(self) -> list[dict[str, Any]]: ...

_world = ...

class _WorldMeta(type):
    @property
    def WORLD(cls) -> Optional[ProcessGroup]: ...
    @WORLD.setter
    def WORLD(cls, pg: Optional[ProcessGroup]):  # -> None:
        ...

class group(metaclass=_WorldMeta): ...

class GroupMember(metaclass=_WorldMeta):
    NON_GROUP_MEMBER = ...

_default_pg_init_method: Optional[str] = ...
STORE_BASED_BARRIER_PREFIX = ...

def get_group_rank(group: ProcessGroup, global_rank: int) -> int: ...
def get_global_rank(group: ProcessGroup, group_rank: int) -> int: ...
def get_process_group_ranks(group: Optional[ProcessGroup]) -> list[int]: ...
def is_mpi_available() -> bool: ...
def is_nccl_available() -> bool: ...
def is_gloo_available() -> bool: ...
def is_ucc_available() -> bool: ...
def is_xccl_available() -> bool: ...
def is_backend_available(backend: str) -> bool: ...
def is_initialized() -> bool: ...
def is_torchelastic_launched() -> bool: ...
def get_backend_config(group: Optional[ProcessGroup] = ...) -> str: ...
def get_backend(group: Optional[ProcessGroup] = ...) -> Backend: ...
def get_default_backend_for_device(device: Union[str, torch.device]) -> str: ...
def get_pg_count() -> int: ...
def get_node_local_rank(fallback_rank: Optional[int] = ...) -> int: ...
@_exception_logger
@_time_logger
def init_process_group(
    backend: Optional[str] = ...,
    init_method: Optional[str] = ...,
    timeout: Optional[timedelta] = ...,
    world_size: int = ...,
    rank: int = ...,
    store: Optional[Store] = ...,
    group_name: str = ...,
    pg_options: Optional[Any] = ...,
    device_id: Optional[Union[torch.device, int]] = ...,
) -> None: ...
def destroy_process_group(group: Optional[ProcessGroup] = ...):  # -> None:

    ...
def get_rank(group: Optional[ProcessGroup] = ...) -> int: ...
def get_world_size(group: Optional[ProcessGroup] = ...) -> int: ...
def isend(
    tensor: torch.Tensor,
    dst: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    tag: int = ...,
    group_dst: Optional[int] = ...,
) -> Optional[Work]: ...
def irecv(
    tensor: torch.Tensor,
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    tag: int = ...,
    group_src: Optional[int] = ...,
) -> Optional[Work]: ...
@_exception_logger
def send(
    tensor: torch.Tensor,
    dst: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    tag: int = ...,
    group_dst: Optional[int] = ...,
) -> None: ...
@_exception_logger
def recv(
    tensor: torch.Tensor,
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    tag: int = ...,
    group_src: Optional[int] = ...,
) -> int: ...

class _IllegalWork(Work):
    def __getattribute__(self, name):  # -> None:
        ...

class _CoalescingManager:
    def __init__(self) -> None: ...
    def append(self, work: Optional[Work] = ...):  # -> None:
        ...
    def wait(self):  # -> None:
        ...

class _TimeEstimator:
    def __init__(self) -> None: ...

def batch_isend_irecv(p2p_op_list: list[P2POp]) -> list[Work]: ...
@_exception_logger
def broadcast(
    tensor: torch.Tensor,
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    async_op: bool = ...,
    group_src: Optional[int] = ...,
):  # -> Work | None:

    ...
@_exception_logger
def all_reduce(tensor, op=..., group=..., async_op=...):  # -> Any | _IllegalWork | Work | None:

    ...
@_exception_logger
@deprecated(
    "`torch.distributed.all_reduce_coalesced` will be deprecated. If you must "
    "use it, please revisit our documentation later at "
    "https://pytorch.org/docs/main/distributed.html#collective-functions",
    category=FutureWarning,
)
def all_reduce_coalesced(tensors, op=..., group=..., async_op=...):  # -> Future[Any] | None:

    ...
@_exception_logger
def reduce(
    tensor: torch.Tensor,
    dst: Optional[int] = ...,
    op=...,
    group: Optional[ProcessGroup] = ...,
    async_op: bool = ...,
    group_dst: Optional[int] = ...,
):  # -> Work | None:

    ...
@_exception_logger
def all_gather_object(object_list, obj, group=...):  # -> None:

    ...
@_exception_logger
def gather_object(
    obj: Any,
    object_gather_list: Optional[list[Any]] = ...,
    dst: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    group_dst: Optional[int] = ...,
):  # -> None:

    ...
@_exception_logger
def send_object_list(
    object_list: list[Any],
    dst: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    device: Optional[torch.device] = ...,
    group_dst: Optional[int] = ...,
    use_batch: bool = ...,
):  # -> None:

    ...
@_exception_logger
def recv_object_list(
    object_list: list[Any],
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    device: Optional[torch.device] = ...,
    group_src: Optional[int] = ...,
    use_batch: bool = ...,
):  # -> int:

    ...
@_exception_logger
def broadcast_object_list(
    object_list: list[Any],
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    device: Optional[torch.device] = ...,
    group_src: Optional[int] = ...,
):  # -> None:

    ...
@_exception_logger
def scatter_object_list(
    scatter_object_output_list: list[Any],
    scatter_object_input_list: Optional[list[Any]] = ...,
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    group_src: Optional[int] = ...,
):  # -> None:

    ...
@_exception_logger
def all_gather(tensor_list, tensor, group=..., async_op=...):  # -> Any | Work | None:

    ...
@_exception_logger
def all_gather_into_tensor(
    output_tensor, input_tensor, group=..., async_op=...
):  # -> Any | _IllegalWork | Work | None:

    ...
@_exception_logger
@deprecated(
    "`torch.distributed.all_gather_coalesced` will be deprecated. If you must use it, "
    "please revisit our documentation later at "
    "https://pytorch.org/docs/main/distributed.html#collective-functions",
    category=FutureWarning,
)
def all_gather_coalesced(output_tensor_lists, input_tensor_list, group=..., async_op=...):  # -> Future[Any] | None:

    ...
@_exception_logger
def gather(
    tensor: torch.Tensor,
    gather_list: Optional[list[torch.Tensor]] = ...,
    dst: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    async_op: bool = ...,
    group_dst: Optional[int] = ...,
):  # -> Work | None:

    ...
@_exception_logger
def scatter(
    tensor: torch.Tensor,
    scatter_list: Optional[list[torch.Tensor]] = ...,
    src: Optional[int] = ...,
    group: Optional[ProcessGroup] = ...,
    async_op: bool = ...,
    group_src: Optional[int] = ...,
):  # -> Work | None:

    ...
@_exception_logger
def reduce_scatter(output, input_list, op=..., group=..., async_op=...):  # -> Work | None:

    ...
@_exception_logger
def reduce_scatter_tensor(output, input, op=..., group=..., async_op=...):  # -> Any | _IllegalWork | Work | None:

    ...
@_exception_logger
def all_to_all_single(
    output, input, output_split_sizes=..., input_split_sizes=..., group=..., async_op=...
):  # -> Any | Work | None:

    ...
@_exception_logger
def all_to_all(output_tensor_list, input_tensor_list, group=..., async_op=...):  # -> Work | None:

    ...
@_exception_logger
def barrier(group: Optional[ProcessGroup] = ..., async_op=..., device_ids=...):  # -> Work | None:

    ...
def monitored_barrier(group: Optional[ProcessGroup] = ..., timeout=..., wait_all_ranks=...):  # -> None:

    ...
@_time_logger
def split_group(
    parent_pg: Optional[ProcessGroup] = ...,
    split_ranks: Optional[list] = ...,
    timeout: Optional[timedelta] = ...,
    pg_options: Optional[Any] = ...,
    group_desc: Optional[str] = ...,
) -> Optional[ProcessGroup]: ...
@_time_logger
def new_group(
    ranks=...,
    timeout=...,
    backend=...,
    pg_options=...,
    use_local_synchronization=...,
    group_desc=...,
    device_id: Optional[torch.device] = ...,
):  # -> None:

    ...
def new_subgroups(
    group_size=..., group=..., timeout=..., backend=..., pg_options=..., group_desc=...
):  # -> tuple[Any | None, list[Any]]:

    ...
def new_subgroups_by_enumeration(
    ranks_per_subgroup_list, timeout=..., backend=..., pg_options=..., group_desc=...
):  # -> tuple[Any | None, list[Any]]:

    ...
