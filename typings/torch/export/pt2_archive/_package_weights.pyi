from collections import UserDict

import torch
from torch.utils._ordered_set import OrderedSet

class TensorProperties:
    def __init__(self, tensor: torch.Tensor) -> None: ...
    def is_complete(self) -> bool:
        """Whether the tensor completely overlaps with its underlying storage"""

class Weights(UserDict):
    """
    A dictionary mapping from weight name to a tuple of (tensor, TensorProperties).
    tensor represents the actual initial value of the weight.
    TensorProperties represents the properties of the weight that are needed to recover the weight.

    We use two separate entries because `tensor` could be a clone of the original weight tensor,
    so it doesn't have the same property as the original weight (such as underlying storage pointer).
    """
    def __init__(self, weight_dict: dict[str, tuple[torch.Tensor, TensorProperties]]) -> None: ...
    def get_weight(self, name: str) -> tuple[torch.Tensor, TensorProperties]: ...
    def get_weight_properties(self, name: str) -> TensorProperties: ...

def get_complete(group: OrderedSet[tuple[str, str]], models_weights: dict[str, Weights]) -> tuple[str, str]:
    """
    `group` is a (model_name, weight_name) tuple.
    `model_weights` is a dictionary mapping from model name to its Weights.

    One of the tensor in `group` must be complete and they must share the
    same underlying storage.

    Returns the name of the complete tensor in the `group`. If multiple
    tensors are complete, returns an arbitrary one.
    """

def group_weights(all_weights: dict[str, Weights]) -> list[OrderedSet[tuple[str, str]]]:
    """
    Group weights that share the same underlying storage.

    Returns a list of sets, each set contains a tuple of (model_name, weight_name).
    """
