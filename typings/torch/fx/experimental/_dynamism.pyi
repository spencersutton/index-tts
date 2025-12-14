from collections.abc import Callable
from typing import Any, TypeAlias, Union

import torch

type KeyPath = tuple[Any, ...]
type NonTensorShapeFn = Callable[[int | float], tuple[Any, ...]]
__all__ = [
    "normalize_source_name",
    "module_to_nested_dict",
    "track_dynamism_across_examples",
    "clone_and_convert_to_meta",
]

def normalize_source_name(name: str) -> str: ...
def module_to_nested_dict(module: torch.nn.Module) -> dict[str, Any]: ...
def track_dynamism_across_examples(example_inputs: list[Any]) -> dict[Any, Any]: ...
def clone_and_convert_to_meta(example_input: Any) -> Any: ...
