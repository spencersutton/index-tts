from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

"""Contains helpers to split tensors into shards."""
TensorT = TypeVar("TensorT")
type TensorSizeFn_T[TensorT] = Callable[[TensorT], int]
type StorageIDFn_T[TensorT] = Callable[[TensorT], Any | None]
MAX_SHARD_SIZE = ...
SIZE_UNITS = ...
logger = ...

@dataclass
class StateDictSplit:
    is_sharded: bool = ...
    metadata: dict[str, Any]
    filename_to_tensors: dict[str, list[str]]
    tensor_to_filename: dict[str, str]
    def __post_init__(self):  # -> None:
        ...

def split_state_dict_into_shards_factory[TensorT](
    state_dict: dict[str, TensorT],
    *,
    get_storage_size: TensorSizeFn_T,
    filename_pattern: str,
    get_storage_id: StorageIDFn_T = ...,
    max_shard_size: int | str = ...,
) -> StateDictSplit: ...
def parse_size_to_int(size_as_str: str) -> int: ...
