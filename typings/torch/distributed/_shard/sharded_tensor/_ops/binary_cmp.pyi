import torch
from torch.distributed._shard.sharded_tensor import _sharded_op_impl

def binary_cmp(cmp_fun, types, args, kwargs=..., process_group=...):  # -> Literal[False]:
    ...
@_sharded_op_impl(torch.equal)
def equal(types, args, kwargs, process_group):  # -> Literal[False]:
    ...
@_sharded_op_impl(torch.allclose)
def allclose(types, args, kwargs, process_group):  # -> Literal[False]:
    ...
