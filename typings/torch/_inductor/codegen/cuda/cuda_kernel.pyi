import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sympy import Expr
from torch._inductor.codegen.cuda.cuda_template import CUDATemplate
from torch._inductor.scheduler import BaseSchedulerNode

from ...autotune_process import CUDABenchmarkRequest
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUDATemplateBuffer,
    IRNode,
    Layout,
    PrimitiveInfoType,
    ShapeAsConstantBuffer,
    TensorBox,
)
from ..common import CSEVariable, Kernel, OpOverrides
from .cuda_template import ArgInfo

log = ...
cexpr = ...
type ValidLayoutSymbols = Literal["M", "N", "K", "B", "lda", "ldb", "ldc", "ldd"]
type ValidLayoutAttrs = Literal["size", "stride"]

@dataclass(frozen=True)
class LayoutArg:
    """LayoutArg(node: torch._inductor.ir.IRNode, symbol: Literal['M', 'N', 'K', 'B', 'lda', 'ldb', 'ldc', 'ldd'], attr: Literal['size', 'stride'], dim: int)"""

    node: IRNode
    symbol: ValidLayoutSymbols
    attr: ValidLayoutAttrs
    dim: int
    def matches(self, node, attr, dim) -> bool: ...

class CUDAKernel(Kernel):
    """Baseclass for CUDA / Cutlass based Kernels"""

    overrides = OpOverrides
    def __init__(self, *args, **kwargs) -> None: ...
    def find_symbol(self, node: IRNode, attr: ValidLayoutAttrs, dim: int) -> str | None: ...
    def find_layout_arg(self, node: IRNode, attr: ValidLayoutAttrs, dim: int) -> LayoutArg | None: ...
    def add_layout_arg(self, symbol: ValidLayoutSymbols, node: IRNode, attr: ValidLayoutAttrs, dim: int): ...
    def init_layout_args(self) -> None: ...
    def get_layout_args(self) -> tuple[Expr | int, ...]: ...
    def get_dynamic_shape_args(self) -> list[Expr | int]: ...
    def get_offset_args(self) -> list[Expr]: ...
    @staticmethod
    def find_ld_idx(node: IRNode) -> int: ...

class CUDATemplateKernel(CUDAKernel):
    """Template kernels defined by CUDA / Cutlass in C++."""

    _EXTRA_CPP_ARGS = ...
    def __init__(self, kernel_name: str, runtime_arg_info: list[ArgInfo], runtime_arg_values: list[Any]) -> None:
        """
        Initializes a new instance of the CUDATemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
    def check_not_null(self, node: IRNode) -> str:
        """Generates code to check that a node is not null."""
    def get_signature(self) -> str: ...
    def def_kernel(
        self, inputs: list[IRNode], outputs: list[IRNode], names_str: str = ..., input_reorder: list[int] | None = ...
    ) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
            additional_size_args: Additional size arguments for epilogue inputs
        """
    def call_kernel(self, name: str, node: CUDATemplateBuffer) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The CUDATemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
    def dtype(self, node: IRNode) -> str | None:
        """Generates code which represents dtype of a given node."""
    def cutlass_dtype(self, node: IRNode, default_dtype=...) -> str | None: ...
    def max_valid_index(self, node: IRNode, default=...): ...
    def ptr(self, node: IRNode) -> str:
        """Generates code which represents pointer of a given node."""
    def size(self, node: IRNode, start_index: int, end_index: int | None = ..., default_value: int = ...) -> str:
        """
        Hook called from template code to get the size of an arg.
        Generates code which represents size of a given node in [start_index, end_index).
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """
    def stride(self, node: IRNode, index: int, default_value: int = ...) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Generates code which represents stride of a given node at index.
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """
    def batch_stride(self, node: IRNode, default_value: int = ...) -> str:
        """
        Hook called from template code to get the batch stride of an arg.
        Returns 0 if batch dim is not present.

        This method assumes that batch stride is the largest stride.
        """
    def row_or_column_stride(self, node: IRNode, default_value: int = ...) -> str:
        """
        Hook called from template code to get the row or column stride of an arg.
        This is required by some CUTLASS 2.X APIs.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].

        TODO: Will add needed args to pass it in if it is dynamic.
        """
    def load(self, name: str, index: Expr, mode: Any = ...) -> CSEVariable:
        """Mock load function for memory planning to optimize allocations properly."""
    def store(self, name: str, index: Expr, value: Any, mode: Any = ...) -> None:
        """Mock store function for memory planning to optimize allocations properly."""

class CUDATemplateCaller(ChoiceCaller):
    """
    CUDATemplateCaller

    This class represents a caller for CUDA template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUDABenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUDATemplateBuffer): The template buffer for the caller.
    """
    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Callable[
            [CUDATemplateBuffer, list[BaseSchedulerNode] | None], tuple[CUDATemplateKernel, functools.partial[str]]
        ],
        bmreq: CUDABenchmarkRequest,
        supports_epilogue_fusion: bool,
        template: CUDATemplate,
        info_kwargs: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None,
        description: str,
    ) -> None: ...
    def precompile(self) -> None: ...
    def benchmark(self, *args, out) -> float: ...
    def call_name(self) -> str: ...
    def kernel_hash_key(self) -> str:
        """Return kernel hash key that does not depend on swizzle."""
    def hash_key(self) -> str:
        """Return kernel hash key that does not depend on swizzle."""
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """
        Information returned here is logged to the autotune log file when that is enabled.

        In general, we should avoid calling this function as it is expensive to compute,
        and can add up very fast.
        """
    def output_node(self) -> TensorBox | ShapeAsConstantBuffer: ...
