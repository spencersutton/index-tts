from torch.distributed._shard.sharding_spec import ShardingSpec

from ._ops import *
from .api import Shard, ShardedTensor

def empty(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def ones(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def zeros(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def full(
    sharding_spec: ShardingSpec,
    size,
    fill_value,
    *,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def rand(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def randn(
    sharding_spec: ShardingSpec,
    *size,
    dtype=...,
    layout=...,
    requires_grad=...,
    pin_memory=...,
    memory_format=...,
    process_group=...,
    init_rrefs=...,
) -> ShardedTensor: ...
def init_from_local_shards(
    local_shards: list[Shard], *global_size, process_group=..., init_rrefs=...
) -> ShardedTensor: ...
def state_dict_hook(module, destination, prefix, local_metadata): ...
def pre_load_state_dict_hook(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
): ...
def custom_sharded_op_impl(func): ...
