from collections.abc import Callable
from typing import Any

import torch
from torch._inductor import ir
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import Buffer, Layout

log = ...

class SubgraphChoiceCaller(ir.ChoiceCaller):
    """
    Represents a Subgraph Autotuning choice, and the subgraph can be any arbitrary
    GraphModule. Compiles the Subgraph down to a module for benchmarking.
    """
    def __init__(
        self, name: str, input_nodes: list[Buffer], layout: Layout, description: str, make_fx_graph: Callable[..., Any]
    ) -> None: ...
    def benchmark(self, *args: list[Any], out: torch.Tensor) -> float: ...
    def hash_key(self) -> str: ...
    def output_node(self) -> ir.TensorBox | ir.ShapeAsConstantBuffer: ...
    def info_dict(self) -> dict[str, Any]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def autoheuristic_id(self) -> str: ...

class SubgraphTemplate(KernelTemplate):
    """
    A template for subgraph evaluation to be used in autotuning.

    This class allows creating customized subgraphs that can be appended
    as choices during the autotuning process, enabling the selection of
    optimal implementations for complex operations.
    """

    index_counter = ...
    def __init__(self, name: str) -> None:
        """
        Initialize a subgraph template.

        Args:
            name: The name of this template
            graph: The FX graph
        """
    def generate(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_fx_graph: Callable[..., Any],
        description: str = ...,
        **kwargs: Any,
    ) -> SubgraphChoiceCaller:
        """
        Generate a SubgraphChoiceCaller instance for autotuning.

        Args:
            input_nodes: List of input nodes to the subgraph
            layout: Memory layout information for the output
            example_inputs: Example tensor inputs used to trace and benchmark the subgraph
            **kwargs: Additional keyword arguments

        Returns:
            SubgraphChoiceCaller: A callable object that can be used for autotuning
        """
