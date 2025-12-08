import pdb
import typing

import torch
from torch._C._distributed_c10d import ProcessGroup as ProcessGroup

from .distributed_c10d import *

log = ...

def is_available() -> bool: ...

if is_available() and not torch._C._c10d_init(): ...
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError
if is_available():
    class _DistributedPdb(pdb.Pdb):
        def interaction(self, *args, **kwargs) -> None: ...

    _breakpoint_cache: dict[int, typing.Any] = ...
    def breakpoint(rank: int = ..., skip: int = ..., timeout_s=...) -> None: ...

else:
    class _ProcessGroupStub: ...
