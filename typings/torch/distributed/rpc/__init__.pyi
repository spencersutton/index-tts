from collections.abc import Generator

import torch
from torch._C._distributed_c10d import Store

from .api import *

__all__ = ["is_available"]
logger = ...
_init_counter = ...
_init_counter_lock = ...

def is_available() -> bool: ...

if is_available() and not torch._C._rpc_init(): ...
if is_available():
    _is_tensorpipe_available = ...
    rendezvous_iterator: Generator[tuple[Store, int, int]]
    def init_rpc(name, backend=..., rank=..., world_size=..., rpc_backend_options=...) -> None:
        """
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd, which immediately makes the current
        process ready to send and receive RPCs.

        Args:
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, colon,
                and/or dash, and must be shorter than 128 characters.
            backend (BackendType, optional): The type of RPC backend
                implementation. Supported values is
                ``BackendType.TENSORPIPE`` (the default).
                See :ref:`rpc-backends` for more information.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions, optional): The options
                passed to the RpcAgent constructor. It must be an agent-specific
                subclass of :class:`~torch.distributed.rpc.RpcBackendOptions`
                and contains agent-specific initialization configurations. By
                default, for all agents, it sets the default timeout to 60
                seconds and performs the rendezvous with an underlying process
                group initialized using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDR`` and
                ``MASTER_PORT`` need to be set properly. See
                :ref:`rpc-backends` for more information and find which options
                are available.
        """
