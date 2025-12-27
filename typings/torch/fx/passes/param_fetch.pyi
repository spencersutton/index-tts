from collections.abc import Callable
from typing import Any

from torch import nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule

__all__ = ["default_matching", "extract_attrs_for_lowering", "lift_lowering_attrs_to_nodes"]

@compatibility(is_backward_compatible=False)
def default_matching(name: str, target_version: int) -> str:
    """
    Default matching method
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

module_fetch_book: dict[type, tuple[int, list[str], Callable[[str, int], str]]] = ...

@compatibility(is_backward_compatible=False)
def extract_attrs_for_lowering(mod: nn.Module) -> dict[str, Any]:
    """
    If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

@compatibility(is_backward_compatible=False)
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """
    Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module.
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
