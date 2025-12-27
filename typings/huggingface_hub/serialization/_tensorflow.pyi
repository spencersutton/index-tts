import tensorflow as tf

from ._base import StateDictSplit

"""Contains tensorflow-specific helpers."""

def split_tf_state_dict_into_shards(
    state_dict: dict[str, tf.Tensor], *, filename_pattern: str = ..., max_shard_size: int | str = ...
) -> StateDictSplit: ...
def get_tf_storage_size(tensor: tf.Tensor) -> int: ...
