import pdb
import typing

import torch
from torch._C._distributed_c10d import ProcessGroup as ProcessGroup

from .distributed_c10d import *

log = ...

def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """

if is_available() and not torch._C._c10d_init(): ...
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError
if is_available():
    class _DistributedPdb(pdb.Pdb):
        """
        Supports using PDB from inside a multiprocessing child process.

        Usage:
        _DistributedPdb().set_trace()
        """
        def interaction(self, *args, **kwargs): ...

    _breakpoint_cache: dict[int, typing.Any] = ...
    def breakpoint(rank: int = ..., skip: int = ..., timeout_s=...):
        """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
            skip (int): Skip the first ``skip`` calls to this breakpoint. Default: ``0``.
        """

else:
    class _ProcessGroupStub: ...
