import dataclasses
import sympy
import torch
from typing import Any, Callable, Optional, Union, TypeAlias
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
    symbol: sympy.Symbol
    def get_name(self) -> str: ...
    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]: ...

CodegenBuffer: TypeAlias = Union[BufferLike, SymbolBuffer]

@dataclasses.dataclass
class TritonKernel:
    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper

def replace_floor_div(expr: sympy.Expr) -> sympy.Expr: ...

class WrapperFxCodegen(PythonWrapperCodegen):
    supports_caching = ...
    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]: ...
    @classmethod
    def create(
        cls,
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = ...,
    ) -> WrapperFxCodegen: ...

@dataclasses.dataclass
class FxConverter:
    lines: list[Line]
    prologue: str = ...
    def __post_init__(self) -> None: ...
    def generate(self) -> torch.fx.GraphModule: ...
