from typing import Any, NamedTuple

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target

__all__ = ["get_size_of_all_nodes", "get_size_of_node", "get_tensor_meta", "replace_target_nodes_with", "size_bytes"]

@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule, old_op: str, old_target: Target, new_op: str, new_target: Target
) -> None:
    """
    Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,
    and updates them to match the new op code and target
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    """
    size_bytes(output_size, total_size)
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

    output_size: int
    total_size: int

@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(fx_module: GraphModule, args: list[torch.Tensor] | None = ...) -> None:
    """
    Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any:
    """
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """
    Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
