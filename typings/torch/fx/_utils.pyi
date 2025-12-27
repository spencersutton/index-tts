import torch

def lazy_format_graph_code(name, gm, maybe_id=..., **kwargs):
    """Returns a LazyString that formats the graph code."""

def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> dict | None:
    """Returns the nn_module_stack of the first call_function node."""

def get_node_context(node, num_nodes=...) -> str:
    """Returns a string of the last num_nodes nodes in the graph."""
