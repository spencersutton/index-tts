import torch.fx
from torch.utils._ordered_set import OrderedSet

from .graph_region_tracker import Node

type UsageIndex = tuple[int, int]
log = ...
last_node_to_additional_deps: dict[Node, OrderedSet[Node]] | None = ...

def apply_graph_deduplication(output_graph) -> dict[str, torch.fx.GraphModule]: ...
