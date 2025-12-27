import contextlib
import dataclasses
import functools
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, NamedTuple, Self

import sympy
import torch
from torch._inductor.codegen.simd import IterationRangesRoot
from torch.utils._ordered_set import OrderedSet

from . import ir
from .codecache import PersistentCache
from .codegen.common import CSEVariable, IndentedBuffer, KernelTemplate, WorkspaceArg
from .codegen.triton import TMACompatibilityChecker, TritonKernel
from .ir import ChoiceCaller, PrimitiveInfoType
from .ops_handler import StoreMode
from .virtualized import V

log = ...
VERIFY: dict[str, Any] = ...
PRINT_AUTOTUNE = ...
DEBUG = ...

class KernelNamespace: ...

extern_kernels = ...

@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""

    input_tensors: list[torch.Tensor]
    output_tensor: torch.Tensor | None
    def unpack(self): ...

@dataclasses.dataclass
class AutotuneArgs:
    """
    During autotuning, we need to pass the same inputs to all choices.
    Note:
        Since we typically have a mix of external choices and triton choices, we create
        two lists of inputs for the same underlying buffers:
        - External inputs (for aten kernels): Include offset for sliced tensors
        - Triton inputs: Use base pointer for sliced tensors, without offset
    """

    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: torch.Tensor | None = ...
    def get_benchmark_tensors(self, extern=...) -> BenchmarkTensors:
        """Returns the inputs and output tensors for a given choice."""
    @classmethod
    def from_choice_args(
        cls,
        example_inputs: list[torch.Tensor],
        example_inputs_extern: list[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: torch.Tensor | None = ...,
    ) -> Self:
        """Factory method to create AutotuneInputs from separate inputs/outputs"""
    def verify(self, **kwargs):
        """Verify the correctness of the benchmarking results"""

class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """

    type HookFn = Callable[[], str]
    def __init__(self, code: str, replacement_hooks: dict[str, HookFn | None]) -> None: ...
    @property
    def code(self) -> str:
        """
        The fully rendered code. Will **error** if any hooks have yet to be
        finalized.
        """
    def finalize_hook(self, hook_key: str, strict: bool = ...) -> None:
        """
        Finalize a hook by name.

        :param strict: If ``True``, raise an error if the hook wasn't found.

        NOTE: Will **error** if the hook has already been finalized.
        """
    def finalize_remaining(self) -> str:
        """
        Finalize the remaining active hooks. This function can be used in cases
        where the caller uses `finalize_hook` rather than `finalize_all`.
        Note: `finalize_all` errors if a hook that has already been finalized
        is attempted to be called again. This function only attempts to
        finalize active hooks.
        """
    def finalize_all(self) -> str:
        """
        Finalize all active hooks.

        NOTE: unlike ``finalize_remaining``, this method will **error** if any
        hook has already been finalized.
        """

@dataclasses.dataclass()
class SubgraphInfo:
    """SubgraphInfo(body: torch._inductor.utils.IndentedBuffer, template_mask: Optional[str] = None, template_out: Optional[str] = None, compute: torch._inductor.utils.IndentedBuffer = <factory>, indexing_code: torch._inductor.utils.IndentedBuffer = <factory>, loads: torch._inductor.utils.IndentedBuffer = <factory>, stores: torch._inductor.utils.IndentedBuffer = <factory>, ops_handler: Optional[torch._inductor.ops_handler.WrapperHandler] = None, range_trees: Optional[list['IterationRangesRoot']] = None, numels: Optional[dict[str, sympy.core.expr.Expr]] = None)"""

    body: IndentedBuffer
    template_mask: str | None = ...
    template_out: str | None = ...
    compute: IndentedBuffer = ...
    indexing_code: IndentedBuffer = ...
    loads: IndentedBuffer = ...
    stores: IndentedBuffer = ...
    ops_handler: V.WrapperHandler | None = ...
    range_trees: list[IterationRangesRoot] | None = ...
    numels: dict[str, sympy.Expr] | None = ...
    def __post_init__(self): ...
    def to_dict(self): ...

class ModificationWrapper(V.WrapperHandler):
    """Handles placeholder substitutions during subgraph processing."""
    def __init__(self, kernel, subgraph_number: int, fixed_inputs: dict[str, Any], mask: str | None) -> None: ...
    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed input."""
    def indirect_indexing(self, index_var: str, size, check, wrap_neg=...):
        """Convert index variable to symbolic form."""
    def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = ...) -> str:
        """
        Currently only supports stores for atomic adds coming from scatter nodes
        This is used by flex_attention's backwards grad for captured buffers, see
        zeros_and_scatter lowering
        """

type RecordedEventsType = list[tuple[str, list[Any], dict[str, Any]]]

class TritonTemplateKernel(TritonKernel):
    """
    A specialized kernel class for Triton templates that handles code generation
    for templated Triton kernels.

    This class extends TritonKernel to provide additional functionality for
    template-based kernel generation, including support for subgraphs, workspace
    arguments, and prologue/epilogue fusion.
    """
    def __init__(
        self,
        kernel_name,
        input_nodes,
        output_node,
        defines,
        num_stages,
        num_warps,
        grid_fn,
        meta,
        call_sizes,
        num_consumer_groups=...,
        num_buffers_warp_spec=...,
        use_jit=...,
        prefix_args=...,
        suffix_args=...,
        epilogue_fn=...,
        subgraphs: list[ir.ComputedBuffer] | None = ...,
        workspace_arg: WorkspaceArg | None = ...,
        prologue_loads_all_inputs=...,
        hint_override: int | None = ...,
    ) -> None: ...
    def input_dependent_preserved_state(self) -> str: ...
    def record_input_dependent_tracked_event(self) -> Callable[..., Any]: ...
    def replay_cached_events(self, events: RecordedEventsType) -> None: ...
    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str): ...
    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str): ...
    def need_numel_args(self): ...
    def estimate_kernel_num_bytes(self):
        """
        Estimate the total number of bytes this kernel takes.
        For in/out nodes, sizes are counted twice: once for reading and
        once for writing.
        """
    def estimate_flops(self) -> int: ...
    def jit_lines(self): ...
    def gen_argdefs(self): ...
    def gen_defines(self): ...
    def def_kernel(self, *argnames):
        """
        Hook called from template code to generate function def and
        needed args.
        """
    def size(self, name: str, index: int):
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
    def stride(self, name, index=...):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
    def modification(
        self, subgraph_number: int, output_name: str | None, mask: str | None = ..., **fixed_inputs
    ) -> str:
        """
        This creates a modification function for a subgraph.
        To use this inside a template, the first argument should specify which subgraph to codegen for

        Args:
            subgraph_number (int): The index of the subgraph in self.subgraphs
            output_name (Optional[str]): The name of the output variable to store the result in
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
        """
    def load_input(
        self,
        input_name: str,
        output_name: str,
        indices: list[Any] | tuple[Any],
        mask: str | None = ...,
        other: float | None = ...,
        indent_width: int = ...,
    ):
        """
        Loads an input and applies any necessary preprocessing or masking.

        Args:
            input_name (str): The name of the input to load.
            indices (Union[List, Tuple]): The index for each dimension of the input.
            val (str): The name of the variable to store the loaded value.
            mask (Optional[str]): An optional mask to use for the load operation.
            other (Optional[Union[float, int]]): The value to use for masked elements. Default is 0.0.
            indent_width (int): The number of spaces to use for indentation.
        """
    def store_output(
        self,
        indices: list[Any] | tuple[Any],
        val: str,
        mask: str | None = ...,
        indent_width: int = ...,
        val_shape: list[str] | None = ...,
    ):
        """
        Stores the final output and appends any epilogue fusions if the buffer hasn't been optimized away.

        Args:
            indices (Union[List, Tuple]): The index for each dimension of the output. The dot product of
                these indices and output strides must match `val`.
            val (str): The value to store.
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
            indent_width (int): The number of spaces to use for indentation. This is used when the call to
                store_output is indented in the kernel definition.
        """
    def render(self, template, kwargs, record_input_dependent_tracked_event=...): ...
    def make_load(self, name, indices, mask):
        """
        Optional helper called from template code to generate the code
        needed to load from an tensor.
        """
    def indexing(
        self,
        index: sympy.Expr,
        *,
        dense_indexing=...,
        copy_shape=...,
        override_mask=...,
        block_ptr=...,
        tma_compatibility_checker: TMACompatibilityChecker | None = ...,
    ):
        """
        Override the default indexing to use our custom mask and force
        dense indexing.
        """
    def codegen_range_tree(self): ...
    def additional_call_args_and_types(self): ...
    def call_kernel(self, name: str, node: ir.IRNode | None = ...): ...
    def kernel_benchmark_extra_args(self) -> list[str]: ...

class GenerateAndLoadResult(NamedTuple):
    """Return type of TritonTemplate.generate_and_load."""

    mod: ModuleType
    extra: str
    input_call_args: tuple[str, ...]
    prologue_supported_inputs: OrderedSet[str]
    kernel_args_sizevars_keys: tuple[sympy.Expr]
    kernel_options: dict[str, Any]

class GeneratedCodeCacheEntry(NamedTuple):
    """GeneratedCodeCacheEntry(code, extra, events)"""

    code: str
    extra: str
    events: list[Any]

class GeneratedCodeCache:
    """
    Cache for generated code. The cache key is a string representation of the input nodes,
    number of stages, number of warps, and call sizes. The cache value is a tuple of the
    generated code, extra code, and events.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def cache_clear(self) -> None: ...
    def make_key(
        self,
        input_nodes: tuple[ir.IRNode],
        num_stages: int,
        num_warps: int,
        call_sizes: Sequence[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Callable[..., Any] | None,
        epilogue_fn_hash: str | None,
        subgraphs: list[ir.Buffer] | None,
        workspace_arg: WorkspaceArg | None,
        layout: ir.Layout,
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        kwargs: dict[str, Any],
        hint_override: int | None = ...,
    ) -> str | None: ...
    def get_entry(self, cache_key: str | None) -> GeneratedCodeCacheEntry | None: ...
    def put_entry(self, cache_key: str | None, code: str, extra: str, events: list[Any]) -> None: ...

class TritonTemplate(KernelTemplate):
    """A Triton template is a template that can be used to generate a Triton kernel."""

    kernel_type: type[Any] = ...
    index_counter = ...
    all_templates: dict[str, TritonTemplate] = ...
    def __init__(
        self,
        name: str,
        grid: Any,
        source: str,
        debug=...,
        cache_codegen_enabled_for_template=...,
        prologue_loads_all_inputs=...,
    ) -> None: ...

    test_cache = ...
    @property
    def uid(self) -> str: ...
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> NotImplementedError | None:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
    def generate_and_load(
        self,
        input_nodes: tuple[ir.IRNode],
        num_stages: int,
        num_warps: int,
        call_sizes: Sequence[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Callable[..., Any] | None,
        epilogue_fn_hash: str | None,
        subgraphs: list[ir.Buffer] | None,
        workspace_arg: WorkspaceArg | None,
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        layout: ir.Layout,
        kwargs: dict[str, Any],
        generate_with_caching,
        hint_override: int | None = ...,
    ) -> GenerateAndLoadResult | None:
        """Generate the python code and load it into the current process"""
    def generate(
        self,
        input_nodes: tuple[ir.IRNode],
        layout: ir.Layout,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = ...,
        num_buffers_warp_spec: int = ...,
        prefix_args: int = ...,
        suffix_args: int = ...,
        epilogue_fn: Callable[..., Any] | None = ...,
        epilogue_fn_hash: str | None = ...,
        subgraphs: list[ir.Buffer] | None = ...,
        mutated_inputs: list[ir.IRNode] | None = ...,
        call_sizes: Sequence[sympy.core.symbol.Symbol] | None = ...,
        workspace_arg: WorkspaceArg | None = ...,
        generate_with_caching=...,
        hint_override: int | None = ...,
        **kwargs,
    ):
        """
        This function generates a TritonTemplateCaller

        Args:
            input_nodes: List of input nodes
            layout: Output layout
            num_stages: Number of stages for triton launch
            num_warps: Number of warps for triton launch
            prefix_args: Number of input nodes to be passed as arguments
            suffix_args: Number of input nodes to be passed as arguments
            epilogue_fn: Optional epilogue function to be called on the output
            subgraphs: Optional subgraphs to be passed as arguments, these will be inlined
                into the triton template string
            mutated_inputs: Optional list of input nodes that are mutated by the kernel, this is helpful
                if you need to return multiple outputs. You can pass them as inputs and mark them as
                being mutated by the kernel.
        """

class ExternKernelChoice:
    def __init__(
        self,
        kernel,
        cpp_kernel=...,
        *,
        name=...,
        has_out_variant=...,
        op_overload=...,
        use_fallback_kernel=...,
        kernel_creator=...,
    ) -> None: ...
    def to_callable(self): ...
    def call_name(self): ...
    @functools.cache
    def hash_key(self): ...
    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=..., **kwargs): ...
    @property
    def uid(self) -> str: ...
    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """
        Maybe generates a new ChoiceCaller and returns it, or None if generation fails.

        kwargs: Additional kwargs to be passed to generate a new ChoiceCaller.
        """
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> NotImplementedError | None: ...

class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def __init__(
        self,
        name,
        input_nodes,
        layout,
        make_kernel_render,
        description,
        bmreq,
        log_info: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None = ...,
        mutated_inputs=...,
        workspace_arg: WorkspaceArg | None = ...,
        allowed_prologue_inps: OrderedSet[str] | None = ...,
        hint_override: int | None = ...,
    ) -> None: ...
    def benchmark(self, *args, out): ...
    def precompile(self): ...
    def call_name(self): ...
    def hash_key(self): ...
    def output_node(self): ...
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def get_make_kernel_render(self): ...
    def autoheuristic_id(self): ...

class ExternKernelCaller(ChoiceCaller):
    def __init__(self, choice: ExternKernelChoice, input_nodes, layout, kwargs=..., *, has_out_variant=...) -> None: ...
    def benchmark(self, *args, out): ...
    def to_callable(self): ...
    def hash_key(self): ...
    def output_node(self): ...
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def autoheuristic_id(self): ...

@functools.cache
def get_mm_log_filename() -> str | None: ...
def append_to_log(filename, data): ...

class DataProcessorChoiceCallerWrapper:
    def __init__(self, wrapped, preprocessor, postprocessor) -> None: ...
    def __getattr__(self, name): ...
    def benchmark(self, *args, out) -> float: ...
    def output_node(self) -> ir.TensorBox: ...

class DataProcessorTemplateWrapper:
    """
    A wrapper class for a kernel template.

    This class together with `DataProcessorChoiceCallerWrapper` provides a convenient way to
    preprocess and postprocess data before and after using the wrapped template. A typical
    usage is to reorder or filter the input nodes in order to match the expected input of other
    kernel choices like a ATen kernel. A more complicated usage is to prepack the weights.
    See the example from :mod:`cpp_gemm_template` for more details.
    """
    def __init__(self, wrapped_template_cls, preprocessor, postprocessor, **kwargs) -> None: ...
    def __getattr__(self, name): ...
    def maybe_append_choice(self, choices, **kwargs): ...
    def generate(self, **kwargs): ...

class ErrorFromChoice(RuntimeError):
    def __init__(self, msg, choice: ChoiceCaller, inputs_str) -> None: ...

class NoValidChoicesError(RuntimeError): ...

@functools.cache
def get_num_workers() -> int: ...
def create_inputs_key(input_nodes) -> str: ...
def create_precompile_key(name: str, inputs_key: str, choices: list[ChoiceCaller]) -> str: ...

type FeedbackFunction = Callable[
    [dict[ChoiceCaller, float], str, list[Any], list[ChoiceCaller], Callable[[], dict[ChoiceCaller, float]]], None
]
type PreprocessingFunction = Callable[[list[ChoiceCaller]], list[ChoiceCaller]]

def filter_choices_by_name_regex(choices: list[ChoiceCaller]) -> list[ChoiceCaller]:
    """Filter choices based on autotune_choice_name_regex config."""

def filter_choices_by_desc_regex(choices: list[ChoiceCaller]) -> list[ChoiceCaller]:
    """Filter choices based on autotune_choice_desc_regex config."""

class AlgorithmSelectorCache(PersistentCache):
    """
    A persistent cache for algorithm selection results used in autotuning of GEMMs
    and convolutions.

    This classes includes precompilation and benchmarking of the kernels.

    The cache is keyed by input characteristics (sizes, strides, dtypes, etc.) but
    doesn't depend on the output layout.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def cache_clear(self) -> None: ...
    def __call__(
        self,
        name,
        choices: list[ChoiceCaller],
        input_nodes,
        layout,
        input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None = ...,
        precompilation_timeout_seconds: int = ...,
        return_multi_template=...,
        best_config_future=...,
    ): ...
    def make_precompile_fn(
        self, choices, name: str, inputs_key: str, precompilation_timeout_seconds: int | None = ...
    ) -> Callable[[], None]:
        """Returns a function that precompiles the given choices."""
    @classmethod
    def get_inputs(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None,
        hint_override: int | None = ...,
    ) -> AutotuneArgs:
        """Factory method to create AutotuneArgs from a list of ChoiceCallers."""
    @classmethod
    def benchmark_choice(cls, choice: ChoiceCaller, autotune_args: AutotuneArgs) -> float: ...
    @classmethod
    def benchmark_choices(
        cls, choices: Sequence[ChoiceCaller], autotune_args: AutotuneArgs
    ) -> dict[ChoiceCaller, float]: ...
    @classmethod
    def benchmark_in_current_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None,
        hint_override: int | None = ...,
    ) -> dict[ChoiceCaller, float]: ...
    @classmethod
    def benchmark_in_sub_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None,
        hint_override: int | None = ...,
    ): ...
    @classmethod
    def make_benchmark_fn(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None,
        hint_override: int | None = ...,
    ): ...
    @staticmethod
    def prescreen_choices(
        choices: list[ChoiceCaller], name: str, inputs_key: str, prescreen_cache: dict[str, OrderedSet[str]]
    ) -> list[ChoiceCaller]:
        """
        Figure out what choices need to be prescreened before autotuning with runtime
        params.

        Prescreening is a process of reducing the number of autotuning for choices with
        runtime params via a two stage autotuning process. First, we fix a set of runtime
        params (here we use swizzle=2) and run autotuning to get a set of candidates.
        Then, we run autotuning again with the candidates and the full set of runtime
        params.

        Since have the concept of runtime params, we need to differentiate between
        choice's hash_key and choice's kernel_hash_key. The former includes information
        like runtime params, while the latter does not. prescreen_cache, if exists, stores
        the set of hash_key that should win the prescreening.

        Right now, only CUTLASS choices have runtime params.
        """
    @staticmethod
    def prune_choices_postscreen(
        choices: list[ChoiceCaller],
        candidate_timings: dict[ChoiceCaller, float],
        name: str,
        inputs_key: str,
        prescreen_cache: dict[str, OrderedSet[str]],
    ) -> list[ChoiceCaller]:
        """Prune the choices after prescreening."""
    @staticmethod
    def log_results(
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict[ChoiceCaller, float],
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: float | None = ...,
        hint_override: int | None = ...,
    ): ...
    @staticmethod
    def benchmark_example_value(node, hint_override: int | None = ...):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
    @staticmethod
    def generate_example_value(size, stride, device, dtype, extra_size, allocation_size=...): ...
    @staticmethod
    def key_of(node):
        """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
    def add_feedback_saver(self, fn: FeedbackFunction): ...
    def clear_feedback_savers(self): ...
    def add_preprocessing_fn(self, fn: PreprocessingFunction): ...
    def clear_preprocessing_fns(self, clear_defaults: bool = ...):
        """
        Clear preprocessing functions.

        Args:
            clear_defaults: If True, clears all functions including defaults.
                           If False, clears only user-added functions and re-registers defaults.
        """

_ALGORITHM_SELECTOR_CACHE: AlgorithmSelectorCache | None = ...

def get_algorithm_selector_cache() -> AlgorithmSelectorCache:
    """Get the global algorithm selector cache, creating it if it doesn't exist."""

def autotune_select_algorithm(*args, **kwargs): ...
def add_feedback_saver(fn: FeedbackFunction): ...
def clear_feedback_savers():
    """Clear all feedback saver functions."""

def add_preprocessing_fn(fn: PreprocessingFunction):
    """
    Add a preprocessing function to be applied to choices before autotuning.

    Preprocessing functions are called sequentially in the order they were registered,
    with each function receiving the output of the previous one. They can filter,
    reorder, transform, or modify the list of choices in any way.

    Args:
        fn: A function that takes a list of ChoiceCaller objects and returns
            a modified list of ChoiceCaller objects.

    Example:
        def my_filter(choices):
            # Filter out choices with certain names
            return [c for c in choices if 'slow' not in c.name.lower()]

        add_preprocessing_fn(my_filter)
    """

def clear_preprocessing_fns(clear_defaults: bool = ...):
    """
    Clear preprocessing functions at module level.

    Args:
        clear_defaults: If True, clears all functions including defaults.
                       If False, clears only user-added functions and re-registers defaults.
    """

def realize_inputs(*args): ...

class SymbolicGridFn:
    """
    Wrapper around a grid function that allows either int or sympy inputs.

        @SymbolicGridFn
        def grid(x, meta, *, cdiv):
            return cdiv(x, meta["BLOCK_X"])
    """
    def __init__(self, fn: Callable[..., tuple[Any, Any, Any]]) -> None: ...
    def __call__(self, *args, **kwargs) -> tuple[int, int, int]: ...
    def sympy_call(self, *args, **kwargs): ...
