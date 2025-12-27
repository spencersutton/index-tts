from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Node

__all__ = ["SourcePartition", "check_subgraphs_connected", "get_source_partitions"]
logger = ...

@compatibility(is_backward_compatible=False)
@dataclass
class SourcePartition:
    """
    SourcePartition(nodes: list[torch.fx.node.Node], source: Any, input_nodes: list[torch.fx.node.Node] = <factory>, output_nodes: list[torch.fx.node.Node] = <factory>, params: list[torch.fx.node.Node] = <factory>)
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

    nodes: list[Node]
    source: Any
    input_nodes: list[Node] = ...
    output_nodes: list[Node] = ...
    params: list[Node] = ...

@compatibility(is_backward_compatible=False)
def get_source_partitions(
    graph: Graph, wanted_sources: list[Any], filter_fn: Callable[[Node], bool] | None = ...
) -> dict[Any, list[SourcePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_sources: List of sources of nodes that were decomposed from this
            source. This can be a function (ex. torch.nn.functional.linear) or a
            leaf module type (ex. torch.nn.Linear).

    Returns:
        Dictionary mapping sources that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        source.

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

@compatibility(is_backward_compatible=False)
def check_subgraphs_connected(subgraph1: SourcePartition, subgraph2: SourcePartition) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
