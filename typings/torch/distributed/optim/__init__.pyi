import torch

from .optimizer import DistributedOptimizer
from .post_localSGD_optimizer import PostLocalSGDOptimizer
from .utils import as_functional_optim
from .zero_redundancy_optimizer import ZeroRedundancyOptimizer

"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters (:class:`~torch.distributed.rpc.RRef`) and runs the
optimizer locally on the workers where the parameters live.  The distributed
optimizer can use any of the local optimizer :ref:`optimizer-algorithms` to
apply the gradients on each worker.
"""
if hasattr(torch._C, "_rpc_init"): ...
__all__ = [
    "DistributedOptimizer",
    "PostLocalSGDOptimizer",
    "ZeroRedundancyOptimizer",
    "as_functional_optim",
]
