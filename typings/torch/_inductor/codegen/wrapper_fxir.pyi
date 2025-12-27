import dataclasses
from collections.abc import Callable
from typing import Any

import sympy
import torch
from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.fx import GraphModule

from .. import ir
from .common import CodegenSymbol
from .wrapper import BufferLike, Line, PythonWrapperCodegen

aten = ...
log = ...

@dataclasses.dataclass
class SymbolBuffer(CodegenSymbol):
    """Represents a sympy.Symbol graph input."""

    symbol: sympy.Symbol
    def get_name(self) -> str: ...
    def get_example(self) -> torch.Tensor | sympy.Symbol: ...

type CodegenBuffer = BufferLike | SymbolBuffer

@dataclasses.dataclass
class TritonKernel:
    """Stores metadata about Triton kernels for use in FX."""

    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper

def replace_floor_div(expr: sympy.Expr) -> sympy.Expr:
    """Replace sympy.floor with FloorDiv."""

class WrapperFxCodegen(PythonWrapperCodegen):
    """Backend to generate wrapper code as an FX IR graph."""

    supports_caching = ...
    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
        """
        Converts the graph module into a runnable function. The default implementation
        is simply an interpreter calling kernels in eager mode. Derived backends can
        override this to do further compilation.
        """
    @classmethod
    def create(
        cls,
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = ...,
    ) -> WrapperFxCodegen: ...

@dataclasses.dataclass
class FxConverter:
    """
    Generates FX IR from Wrapper IR. As each instance is only meant to be used once, the
    input and output code are stored as attributes.
    """

    lines: list[Line]
    prologue: str = ...
    def __post_init__(self) -> None: ...
    def generate(self) -> torch.fx.GraphModule:
        """Main entrypoint for FX codegen."""
