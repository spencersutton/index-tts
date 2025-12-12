import threading
from collections.abc import Sequence
from typing import Callable, Optional, Union
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import OpInfo, OpSchema, OutputSharding, RuntimeSchemaInfo, StrategyType

aten = ...

class LocalLRUCache(threading.local):
    def __init__(self, user_function: Callable) -> None: ...
    def __call__(self, *args, **kwargs) -> object: ...
    def cache_info(self):  # -> _CacheInfo:
        ...

class ShardingPropagator:
    def __init__(self) -> None: ...
    def register_sharding_prop_rule(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[OpSchema], OutputSharding],
        schema_info: Optional[RuntimeSchemaInfo] = ...,
    ):  # -> None:

        ...
    def register_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_func: Callable[[OpSchema], StrategyType],
        schema_info: Optional[RuntimeSchemaInfo] = ...,
    ):  # -> None:

        ...
    def propagate_tensor_meta(self, op_schema: OpSchema) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]: ...
    def propagate(self, op_info: OpInfo) -> None: ...
    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding: ...
