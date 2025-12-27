import contextlib
import dataclasses
from collections.abc import Callable
from typing import Any

import sympy
from torch._inductor.codegen.common import CSEVariable, IndentedBuffer, Kernel
from torch._inductor.ir import Buffer
from torch._inductor.ops_handler import StoreMode
from torch._inductor.virtualized import V

MAIN_SUFFIX = ...
log = ...
kernel_code_log = ...

class CuteDSLKernelWrapper:
    """Wrapper to provide .run() interface for CuteDSL kernels"""
    def __init__(self, kernel_fn: Callable[..., Any], kernel_path: str | None = ...) -> None: ...
    def run(self, *args, stream=..., **kwargs):
        """
        Execute the CuteDSL kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """

@dataclasses.dataclass
class CuteDSLSubgraphInfo:
    """Minimal subgraph info for CuteDSL kernels."""

    body: IndentedBuffer
    template_mask: str | None = ...
    template_out: str | None = ...
    def to_dict(self): ...

class CuteDSLTemplateKernel(Kernel):
    """
    Template kernel implementation for CuteDSL (CUTLASS Python DSL).
    Handles code generation and argument management for CuteDSL CUDA kernels.
    Provides CuteDSL-specific functionality for tensor conversion and kernel configuration.
    """
    def __init__(
        self, kernel_name: str, input_nodes: list[Buffer], output_node: Buffer, subgraphs: list[Buffer] | None = ...
    ) -> None: ...
    def gen_imports(self) -> str:
        """Generate common imports for CuteDSL templates."""
    def gen_defines(self, **kwargs) -> str:
        """Generate CuteDSL parameter definitions from kwargs, similar to Triton's gen_defines."""
    def render(self, template, **kwargs): ...
    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        """Set the active subgraph body for template processing."""
    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str):
        """Create a new subgraph body for template processing."""
    def def_kernel(self, *argnames):
        """Define kernel function signature for CuteDSL templates."""
    def get_output(self):
        """Get the actual argument name for the output buffer."""
    def call_kernel(self, name: str, node=...):
        """Call the kernel function. Simplified version of TritonTemplateKernel.call_kernel."""
    def modification(
        self, subgraph_number: int, output_name: str | None, mask: str | None = ..., **fixed_inputs
    ) -> str:
        """Generate CuteDSL code for a subgraph modification."""

class ModificationWrapperCuteDSL(V.WrapperHandler):
    """
    Wrapper handler that enables CuteDSL code generation during subgraph modifications.

    This class sits between the PyTorch IR and CuteDSL code generation, providing:
    1. Operation substitution: converts PyTorch ops to CuteDSL equivalents via CuteDSLOpOverrides
    2. Placeholder handling: resolves fixed_inputs during template processing
    3. Limited operation support: currently restricted to pointwise operations
    """
    def __init__(self, kernel, subgraph_number: int, fixed_inputs: dict[str, Any], mask: str | None) -> None: ...
    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed(template args) input for CuteDSL."""
    def indirect_indexing(self, index_var: str, size, check, wrap_neg=...):
        """Convert index variable to symbolic form."""
    def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = ...) -> str: ...
