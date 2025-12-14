import logging
import pdb
import sys
import traceback
import typing
from datetime import timedelta

import torch
from torch._C._distributed_c10d import (
    _DEFAULT_FIRST_BUCKET_BYTES,
    BuiltinCommHookType,
    DebugLevel,
    FileStore,
    GradBucket,
    Logger,
    PrefixStore,
    Reducer,
    Store,
    TCPStore,
    _broadcast_coalesced,
    _compute_bucket_assignment_by_size,
    _ControlCollectives,
    _make_nccl_premul_sum,
    _register_builtin_comm_hook,
    _register_comm_hook,
    _StoreCollectives,
    _test_python_store,
    _verify_params_across_processes,
    get_debug_level,
    set_debug_level,
    set_debug_level_from_env,
)
from torch._C._distributed_c10d import (
    Backend as _Backend,
)
from torch._C._distributed_c10d import (
    ProcessGroup as ProcessGroup,
)
from torch._C._distributed_c10d import (
    Work as _Work,
)

from .device_mesh import DeviceMesh, init_device_mesh
from .distributed_c10d import *
from .distributed_c10d import (
    _all_gather_base,
    _coalescing_manager,
    _CoalescingManager,
    _create_process_group_wrapper,
    _get_process_group_name,
    _rank_not_in_group,
    _reduce_scatter_base,
    _time_estimator,
    get_node_local_rank,
)
from .remote_device import _remote_device
from .rendezvous import _create_store_from_options, register_rendezvous_handler, rendezvous

log = ...

def is_available() -> bool: ...

if is_available() and not torch._C._c10d_init(): ...
DistError = ...
DistBackendError = ...
DistNetworkError = ...
DistStoreError = ...
QueueEmptyError = ...
if is_available():
    class _DistributedPdb(pdb.Pdb):
        def interaction(self, *args, **kwargs):  # -> None:
            ...

    _breakpoint_cache: dict[int, typing.Any] = ...
    def breakpoint(rank: int = ..., skip: int = ..., timeout_s=...):  # -> None:

        ...

else:
    class _ProcessGroupStub: ...
