import torch
from typing import Any, Callable, Union, TypeAlias

KeyPath: TypeAlias = tuple[Any, ...]
NonTensorShapeFn: TypeAlias = Callable[[Union[int, float]], tuple[Any, ...]]
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
