import dataclasses
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, TypeVar, Union

import sympy
import torch
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch.utils._ordered_set import OrderedSet

from ...utils._sympy.symbol import SymT
from ...utils._sympy.value_ranges import ValueRanges
from ..ir import IRNode
from ..scheduler import BaseSchedulerNode, Scheduler
from ..utils import cache_on_self
from ..virtualized import ReductionType, StoreMode
from .common import CSE, BlockShapeType, CSEVariable, IndentedBuffer, OpOverrides, PythonPrinter
from .simd import IterationRanges, IterationRangesEntry, IterationRangesRoot, SIMDKernel, SIMDScheduling
from .simd_kernel_features import SIMDKernelFeatures

_T = TypeVar("_T")
log = ...
perf_hint_log = ...
schedule_log = ...
fusion_log = ...
async_compile = ...

class OpDtypeSupport:
    """
    Some Triton ops such as libdevice and tl.math only support float32 and float64.
    This class records which dtypes are supported by specific IR ops.
    """

    supported_dtypes: dict[str, OrderedSet[torch.dtype]] = ...
    convert_outputs: dict[str, bool] = ...
    @classmethod
    def register_upcast(cls, func: Callable[..., str], convert_output: bool) -> None: ...

@lru_cache(None)
def gen_attr_descriptor_import() -> str:
    """
    import AttrsDescriptor if the triton version is new enough to have this
    class defined.
    """

@lru_cache(None)
def gen_common_triton_imports() -> str: ...

class TritonSymbols:
    """Stores sympy.Symbol instances and constants associated with triton codegen."""

    reduction_types = ...
    block_types = ...
    block_offsets = ...
    block_sizes = ...
    @classmethod
    def get_block_size(cls, tree: IterationRanges) -> sympy.Symbol: ...
    @classmethod
    def get_block_offset(cls, tree: IterationRanges) -> sympy.Symbol: ...

@dataclasses.dataclass
class IndexingOptions:
    """IndexingOptions(index_str: 'str', mask_vars: 'OrderedSet[str]', expand_str: 'Optional[str]', _has_rindex: 'bool', index: 'sympy.Expr', expand_shape: 'Optional[Sequence[Union[int, str]]]')"""

    index_str: str
    mask_vars: OrderedSet[str]
    expand_str: str | None
    _has_rindex: bool
    index: sympy.Expr
    expand_shape: Sequence[int | str] | None
    def has_mask(self) -> bool: ...
    def has_indirect(self) -> bool: ...
    def has_rindex(self) -> bool: ...
    def has_tmpmask(self) -> bool: ...
    def has_rmask(self) -> bool: ...
    @property
    def mask_str(self) -> str: ...

@dataclasses.dataclass
class BlockDescriptorOptions:
    """
    This is a base class that describes a block descriptor used in Triton kernels.
    It can be used to create either a tensor descriptor (with TensorDescriptorOptions)
    or a block pointer (with BlockPtrOptions).
    """

    params: BlockParameters
    constant_offset: sympy.Expr
    order: list[int]
    mask_vars: OrderedSet[str]
    broadcast_shape: Sequence[sympy.Expr]
    broadcasting_dims: list[bool]
    final_shape: Sequence[sympy.Expr]
    _boundary_check: list[int] | None = ...
    @property
    def shape(self) -> list[sympy.Expr]: ...
    @property
    def block_shape(self) -> list[sympy.Expr]: ...
    @property
    def strides(self) -> list[sympy.Expr]: ...
    @property
    def offsets(self) -> list[sympy.Expr]: ...
    @classmethod
    def create(
        cls,
        *,
        params: BlockParameters,
        constant_offset: sympy.Expr,
        range_trees: list[IterationRangesRoot],
        mask_vars: OrderedSet[str],
        get_max_block: Callable[[str], int],
    ) -> BlockDescriptorOptions:
        """Helper to create a BlockDescriptorOptions instance"""
    def replace_offset(self, expr: sympy.Expr, replacement: sympy.Expr, symt: SymT) -> sympy.Expr:
        """Replaces instances of {symt}_offset with the new expression."""
    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr: ...
    def compute_boundary_check(
        self, get_max_block: Callable[[str], int], range_trees: list[IterationRangesRoot]
    ) -> None:
        """List of indices to pass to tl.load(boundary_check=...)"""
    def boundary_check(self) -> list[int]: ...
    def has_indirect(self) -> bool: ...
    def has_rindex(self) -> bool: ...
    def has_rmask(self) -> bool: ...
    def has_tmpmask(self) -> bool: ...
    def has_mask(self) -> bool: ...
    def codegen_broadcast_and_reshape(
        self, value: str, initial_shape: Sequence[sympy.Expr], final_shape: Sequence[sympy.Expr], allow_implicit: bool
    ) -> str:
        """
        Generate a broadcast and a reshape for the block descriptor.
        This restores stride-0 dimensions which were removed from the block descriptor.
        """

@dataclasses.dataclass
class TensorDescriptorOptions(BlockDescriptorOptions):
    """TensorDescriptorOptions(params: 'BlockParameters', constant_offset: 'sympy.Expr', order: 'list[int]', mask_vars: 'OrderedSet[str]', broadcast_shape: 'Sequence[sympy.Expr]', broadcasting_dims: 'list[bool]', final_shape: 'Sequence[sympy.Expr]', _boundary_check: 'Optional[list[int]]' = None)"""
    def format(self, name: str, roffset=...) -> str:
        """
        Codegen a call to tl.make_tensor_descriptor()

        Args:
            name: variable name for pointer
            roffset: unused, but kept for compatibility with BlockPtrOptions.format()

        Returns:
            "tl.make_tensor_descriptor(...)"
        """

@dataclasses.dataclass
class BlockPtrOptions(BlockDescriptorOptions):
    """BlockPtrOptions(params: 'BlockParameters', constant_offset: 'sympy.Expr', order: 'list[int]', mask_vars: 'OrderedSet[str]', broadcast_shape: 'Sequence[sympy.Expr]', broadcasting_dims: 'list[bool]', final_shape: 'Sequence[sympy.Expr]', _boundary_check: 'Optional[list[int]]' = None)"""
    def replace_offset(self, expr: sympy.Expr, replacement: sympy.Expr, symt: SymT) -> sympy.Expr:
        """Replaces instances of {symt}_offset with the new expression."""
    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr: ...
    def format(self, name: str, roffset=...) -> str:
        """
        Codegen a call to tl.make_block_ptr()

        Args:
            name: variable name for pointer
            roffset: should rn_offset be included in offsets=..., for use with tl.advance()

        Returns:
            "tl.make_block_ptr(...)"
        """
    def advance_roffset(self, symt: SymT) -> sympy.Expr:
        """
        Codegen string to pass to tl.advance(name, ...).

        Advance is the difference between offsets in each loop iteration.
        To compute it, we replace rN_offset with multiples of RN_BLOCK.
        Since we expect rN_offset to vary in range(0, rN_numel, RN_BLOCK), the first
        iteration has rN_offset=0, while the second has rN_offset=RN_BLOCK.
        """

def triton_reshape(value: str, old_shape: Sequence[sympy.Expr], new_shape: Sequence[sympy.Expr]) -> str:
    """Workaround https://github.com/triton-lang/triton/issues/2836"""

class TritonPrinter(PythonPrinter): ...

texpr = ...

def triton_compute_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type and upcast [b]float16 to float32"""

def triton_store_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type, with fix for storing tl.bool"""

def upcast_acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Implicit upcasts used for Triton reduction types"""

def triton_acc_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type, with reduction upcasts"""

def low_precision_fp(dtype: torch.dtype) -> bool: ...
def low_precision_fp_var(var: CSEVariable | Any) -> bool: ...

class TritonCSEVariable(CSEVariable):
    def __init__(
        self, name: str, bounds: ValueRanges[Any], dtype: torch.dtype, shape: BlockShapeType = ...
    ) -> None: ...
    def update_on_args(self, name, args, kwargs): ...

def get_dtype_handler() -> DtypePropagationOpsHandler: ...
def maybe_upcast_float32(convert_output: bool = ...) -> Callable[[_T], _T]:
    """
    Codegen helper to upcast arguments to float32, depending on the config and dtype.
    This decorates tl.math/libdevice codegen functions.
    """

class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    _LOG_2_E = ...
    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: torch.dtype | None = ..., use_compute_types=...): ...
    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype): ...
    @classmethod
    def constant(cls, value, dtype): ...
    @staticmethod
    @maybe_upcast_float32()
    def abs(x): ...
    @staticmethod
    def truediv(x, y): ...
    @staticmethod
    def mod(x, y): ...
    @staticmethod
    @maybe_upcast_float32()
    def exp(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def exp2(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def expm1(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def sqrt(x): ...
    @staticmethod
    def relu(x): ...
    @staticmethod
    def minimum(a, b): ...
    @staticmethod
    def maximum(a, b): ...
    @staticmethod
    def where(a, b, c): ...
    @staticmethod
    def inline_asm_elementwise(*inputs, asm, constraints=..., dtype=..., is_pure=..., pack=...): ...
    @staticmethod
    @maybe_upcast_float32()
    def cos(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def sin(x): ...
    @classmethod
    def index_expr(cls, expr, dtype): ...
    @staticmethod
    def masked(mask, body, other): ...
    @staticmethod
    @maybe_upcast_float32()
    def lgamma(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def erf(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def cosh(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def sinh(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def acos(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def acosh(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def asin(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def asinh(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def atan2(x, y): ...
    @staticmethod
    @maybe_upcast_float32()
    def atan(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def atanh(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def copysign(x, y): ...
    @staticmethod
    @maybe_upcast_float32()
    def erfc(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def erfinv(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def hypot(x, y): ...
    @staticmethod
    @maybe_upcast_float32()
    def log10(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def log2(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def nextafter(x, y): ...
    @staticmethod
    def logical_and(a, b): ...
    @staticmethod
    def logical_not(a): ...
    @staticmethod
    def logical_or(a, b): ...
    @staticmethod
    def logical_xor(a, b): ...
    @staticmethod
    def bitwise_and(a, b): ...
    @staticmethod
    def bitwise_not(a): ...
    @staticmethod
    def bitwise_or(a, b): ...
    @staticmethod
    def bitwise_xor(a, b): ...
    @staticmethod
    def bitwise_left_shift(a, b): ...
    @staticmethod
    def bitwise_right_shift(a, b): ...
    @staticmethod
    def rand(seed, offset): ...
    @staticmethod
    def randn(seed, offset): ...
    @staticmethod
    def randint64(seed, offset, low, high): ...
    @staticmethod
    def load_seed(name, offset): ...
    @staticmethod
    @maybe_upcast_float32()
    def rsqrt(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def log1p(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def tan(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def tanh(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def sigmoid(x): ...
    @staticmethod
    def signbit(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def fmod(a, b): ...
    @staticmethod
    @maybe_upcast_float32()
    def pow(a, b): ...
    @staticmethod
    @maybe_upcast_float32()
    def log(x): ...
    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    def isinf(x): ...
    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    def isnan(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def round(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def floor(x): ...
    @staticmethod
    def floordiv(a, b): ...
    @staticmethod
    def sign(x): ...
    @staticmethod
    @maybe_upcast_float32()
    def trunc(x): ...
    @staticmethod
    def truncdiv(a, b): ...
    @staticmethod
    @maybe_upcast_float32()
    def ceil(x): ...

class TritonKernelOverrides(TritonOverrides):
    """
    Map element-wise ops to Triton within a TritonKernel

    Unlike TritonOverrides, these assume the code is going to be inserted into
    the body of the main triton kernel and so it may use indexing and mask
    variables which are assumed to already be defined in the current scope.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def constant(cls, value, dtype): ...
    @classmethod
    def index_expr(cls, expr, dtype): ...
    @staticmethod
    def masked(mask, body, other): ...
    @staticmethod
    def load_seed(name, offset): ...
    @staticmethod
    def frexp(x): ...
    @staticmethod
    def device_assert_async(cond, msg): ...

class HelperFunctions:
    """An ordered set of helper functions."""

    _templates_seen: dict[str, str]
    finalized_helpers: list[str]
    def __init__(self) -> None: ...
    def add(self, template_code: str, *, base_name=...) -> str:
        """
        This accepts a function definition with the function name
        left as a format specifier e.g.

            @triton.jit
            def {name}(arg0, arg1):
                return arg0 + arg1

        We add the templated code to the function set and return the name
        assigned to that function.
        """
    def __iter__(self): ...
    def __getitem__(self, idx): ...

@dataclasses.dataclass
class BlockParameters:
    """Class representing ND block dimensions, for block pointer analysis."""

    shape: list[sympy.Expr] = ...
    block_shape: list[sympy.Expr] = ...
    strides: list[sympy.Expr] = ...
    offsets: list[sympy.Expr] = ...
    def __add__(self, other: BlockParameters) -> BlockParameters:
        """Concatenates block parameters."""

class CooperativeReductionWorkspaceCache:
    """
    The scratch space used for cooperative reductions can be reused
    after two reduction loops.  This keeps track of what can be reused.
    """
    def __init__(self, args) -> None: ...
    def allocate(self, nbytes: sympy.Expr): ...
    def on_loop_end(self): ...
    def increment_store_count(self): ...

@dataclasses.dataclass
class FixedTritonConfig:
    """FixedTritonConfig(config: 'dict[str, int]')"""

    config: dict[str, int]
    def __getitem__(self, item): ...
    def __contains__(self, item) -> bool: ...

class TritonCSE(CSE[TritonCSEVariable, Union[str, tuple[str, str]]]):
    """
    Subclasses CSE to apply the current load mask to the cache key to avoid CSEing
    variables across separate masked blocks.
    """
    def augment_key(self, cache_key: str) -> str | tuple[str, str]: ...

@dataclasses.dataclass
class TMACompatibilityChecker:
    """Checks if the TMA API can be used for load / store triton operations."""

    kernel: TritonKernel
    dtype: torch.dtype
    for_store: bool
    def __post_init__(self): ...
    def can_use_tma(self) -> bool: ...
    def are_block_parameters_compatible(self, block_params: BlockParameters) -> bool:
        """Check if the block parameters are valid for TMA."""

@dataclasses.dataclass
class TMACompatibilityChecker:
    """Checks if the TMA API can be used for load / store triton operations."""

    kernel: TritonKernel
    dtype: torch.dtype
    for_store: bool
    def __post_init__(self): ...
    def can_use_tma(self) -> bool: ...
    def are_block_parameters_compatible(self, block_params: BlockParameters) -> bool:
        """Check if the block parameters are valid for TMA."""

class TritonKernel(SIMDKernel[TritonCSEVariable]):
    """
    A class to represent a triton kernel and helpers to generate
    triton kernel programmatically
    """

    overrides = TritonKernelOverrides
    helper_functions: HelperFunctions
    kexpr: Callable[[sympy.Expr], str] = ...
    allow_block_ptr = ...
    tma_compatibility_checker_cls = TMACompatibilityChecker
    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread=...,
        optimize_mask=...,
        fixed_config: FixedTritonConfig | None = ...,
        hint_override: int | None = ...,
        **kwargs,
    ) -> None: ...
    def dtype_to_str(self, dtype: torch.dtype) -> str: ...
    def should_use_cooperative_reduction(self) -> bool: ...
    def init_cooperative_reduction(self):
        """One time setup code for cooperative reductions."""
    def init_cooperative_reduction_mask(self): ...
    def codegen_range_tree(self): ...
    def need_numel_args(self):
        """
        Indicate whether we need provide numel as arguments for the generated
        kernel calls in the benchmark.

        Should be true for pointwise/reduction kernels but false for triton
        matmul kernels.
        """
    def should_use_persistent_reduction(self) -> bool: ...
    def want_no_x_dim(self): ...
    @property
    def assert_function(self) -> str: ...
    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=...,
        dense_indexing=...,
        override_mask=...,
        block_ptr=...,
        tma_compatibility_checker: TMACompatibilityChecker | None = ...,
    ):
        """Compute the index and mask to pass to tl.load() or tl.store()"""
    def codegen_block_ptr(
        self, name: str, var: str, indexing: BlockPtrOptions | TensorDescriptorOptions, other=...
    ) -> tuple[str, str]: ...
    def codegen_block_ptr_store_line(self, name, indexing, block_ptr, value, other=...): ...
    def check_bounds(self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool): ...
    def get_load_buffer(self, indexing): ...
    def load(self, name: str, index: sympy.Expr):
        """Load from the memory location 'name', offset by some indexing expression 'index'."""
    def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = ...) -> None: ...
    def guard_cooperative_store(self, name, buffer):
        """
        For cooperative reductions only one thread block should write out the result.
        We rotate which thread block does each write for better parallelism
        """
    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = ...,
        sorter_indices: CSEVariable | None = ...,
    ) -> CSEVariable:
        """See [Note: Inductor bucketize op]"""
    def reduction_resize(self, value) -> str: ...
    def reduction_resize_and_shape(self, value, shape) -> tuple[str, BlockShapeType]: ...
    def reduction_collapse_dims(self, buffer, value: CSEVariable, dtype: torch.dtype) -> CSEVariable:
        """Reshape to RBLOCK, collapsing all reduction dims."""
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]: ...
    def welford_reduce(self, result_var, reduction_type, value, where_cond, acc_type, dtype):
        """Helper to codegen a welford reduction"""
    def welford_reduce_final_reduction(
        self, buffer, result_mean, result_m2, result_weight, mean, m2, weight, dim, dtype
    ):
        """Helper to codegen call to triton_helpers.welford"""
    def online_softmax_reduce_final_reduction(self, buffer, result_max, result_sum, peer_max, peer_sum, dim, dtype): ...
    def max_rsplit(self): ...
    def codegen_cooperative_reduction_peer_combine(self, result_var, dtype, default_val) -> CSEVariable:
        """
        Generate code to save a [XBLOCK, RSPLIT] temporary workspace, where each thread block writes a different
        column.  After the barrier, every thread block loads the completed value so that it can compute the final
        value independently.
        """
    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable | tuple[CSEVariable, ...]): ...
    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[[tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        """Perform an associative scan on 'values'."""
    def sort(
        self, dtypes: tuple[torch.dtype, ...], values: tuple[CSEVariable, ...], stable: bool, descending: bool
    ) -> tuple[CSEVariable, ...]: ...
    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
    def kernel_benchmark_extra_args(self) -> list[str]: ...
    def codegen_kernel_benchmark(self, num_gb): ...
    def imports_for_benchmark_kernel(self): ...
    @staticmethod
    def inductor_meta_common(): ...
    def codegen_kernel(self, name=...) -> str:
        """
        Convert the TritonKernel from Inductor SIMD IR to triton code, including inductor triton heuristics, imports,
        metadata, and benchmarking infra.
        """
    @staticmethod
    def has_persistent_RBLOCK(rnumel): ...
    def codegen_static_numels(self, code):
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        r0_numel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
    def add_numel_to_call_args(self, name, call_args, arg_types): ...
    def call_kernel(self, name: str, node: IRNode | None = ...): ...
    def codegen_nan_check(self) -> None: ...
    def create_cse_var(self, *args, **kwargs) -> TritonCSEVariable: ...
    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry): ...
    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str: ...
    def iteration_ranges_scalar_code(self, entry: IterationRangesRoot, value: Any) -> str: ...
    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str: ...
    def needs_yz_grid_overflow(self, entry: IterationRangesRoot) -> bool: ...
    def max_block(self, prefix: str) -> int: ...
    def filter_masks(self, mask_vars: OrderedSet[str]) -> None: ...
    @cache_on_self
    def get_reduction_prefixes(self) -> list[str]: ...
    def codegen_reduction_numels(self, buffer: IndentedBuffer) -> None:
        """Generates code that flattens ND reduction numels, block sizes, etc. into 1D."""
    def codegen_reduction_indices(self, buffer: IndentedBuffer) -> None:
        """Generates code that converts ND reduction indices into linear indices."""
    def iteration_ranges_codegen_header(self, entry: IterationRangesRoot, code: IndentedBuffer) -> None: ...

class TritonScheduling(SIMDScheduling):
    kernel_type: type[Any] = ...
    backend_features = ...
    def __init__(self, scheduler: Scheduler | None) -> None: ...
    @classmethod
    def get_backend_features(cls, device: torch.device): ...
    def codegen_comment(self, node_schedule): ...
    def define_kernel(self, src_code, node_schedule, kernel): ...
    def benchmark_fused_nodes(self, nodes, n_spills_threshold=...) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
    def benchmark_codegened_module(
        self, mod, n_spills_threshold=..., node_names: OrderedSet[str] | None = ...
    ) -> tuple[float, str]:
        """Benchmark an already compiled module"""
    def create_kernel_choices(
        self, kernel_features: SIMDKernelFeatures, kernel_args: list[Any], kernel_kwargs: dict[str, Any]
    ) -> list[TritonKernel]: ...
    def add_multi_kernel_choices(
        self, kernel: TritonKernel, kernel_args: list[Any], kernel_kwargs: dict[str, Any]
    ) -> list[TritonKernel]: ...
    def benchmark_combo_kernel(self, node_list): ...

def debug_triton_code(node: BaseSchedulerNode) -> list[str]: ...
