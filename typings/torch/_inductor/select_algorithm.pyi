import contextlib
import dataclasses
import functools
import sympy
import torch
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Callable, NamedTuple, Optional, TYPE_CHECKING, Union, TypeAlias
from typing_extensions import Self
from torch.utils._ordered_set import OrderedSet
from . import ir
from .codecache import PersistentCache
from .codegen.common import CSEVariable, IndentedBuffer, KernelTemplate, WorkspaceArg
from .codegen.triton import TMACompatibilityChecker, TritonKernel
from .ir import ChoiceCaller, PrimitiveInfoType
from .ops_handler import StoreMode
from .virtualized import V
from torch._inductor.codegen.simd import IterationRangesRoot

log = ...
VERIFY: dict[str, Any] = ...
PRINT_AUTOTUNE = ...
DEBUG = ...
if TYPE_CHECKING: ...

class KernelNamespace: ...

extern_kernels = ...

@dataclasses.dataclass
class BenchmarkTensors:
    input_tensors: list[torch.Tensor]
    output_tensor: Optional[torch.Tensor]
    def unpack(self):  # -> tuple[list[Tensor], Tensor | None]:
        ...

@dataclasses.dataclass
class AutotuneArgs:
    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: Optional[torch.Tensor] = ...
    def get_benchmark_tensors(self, extern=...) -> BenchmarkTensors: ...
    @classmethod
    def from_choice_args(
        cls,
        example_inputs: list[torch.Tensor],
        example_inputs_extern: list[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: Optional[torch.Tensor] = ...,
    ) -> Self: ...
    def verify(self, **kwargs):  # -> None:

        ...

class PartialRender:
    HookFn: TypeAlias = Callable[[], str]
    def __init__(self, code: str, replacement_hooks: dict[str, Optional[HookFn]]) -> None: ...
    @property
    def code(self) -> str: ...
    def finalize_hook(self, hook_key: str, strict: bool = ...) -> None: ...
    def finalize_remaining(self) -> str: ...
    def finalize_all(self) -> str: ...

@dataclasses.dataclass()
class SubgraphInfo:
    body: IndentedBuffer
    template_mask: Optional[str] = ...
    template_out: Optional[str] = ...
    compute: IndentedBuffer = ...
    indexing_code: IndentedBuffer = ...
    loads: IndentedBuffer = ...
    stores: IndentedBuffer = ...
    ops_handler: Optional[V.WrapperHandler] = ...
    range_trees: Optional[list[IterationRangesRoot]] = ...
    numels: Optional[dict[str, sympy.Expr]] = ...
    def __post_init__(self):  # -> None:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

class ModificationWrapper(V.WrapperHandler):
    def __init__(self, kernel, subgraph_number: int, fixed_inputs: dict[str, Any], mask: Optional[str]) -> None: ...
    def load(self, name: str, index: sympy.Expr): ...
    def indirect_indexing(self, index_var: str, size, check, wrap_neg=...): ...
    def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = ...) -> str: ...

RecordedEventsType: TypeAlias = list[tuple[str, list[Any], dict[str, Any]]]

class TritonTemplateKernel(TritonKernel):
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
        subgraphs: Optional[list[ir.ComputedBuffer]] = ...,
        workspace_arg: Optional[WorkspaceArg] = ...,
        prologue_loads_all_inputs=...,
        hint_override: Optional[int] = ...,
    ) -> None: ...
    def input_dependent_preserved_state(self) -> str: ...
    def record_input_dependent_tracked_event(self) -> Callable[..., Any]: ...
    def replay_cached_events(self, events: RecordedEventsType) -> None: ...
    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):  # -> Generator[None, Any, None]:
        ...
    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str):  # -> Generator[None, Any, None]:
        ...
    def need_numel_args(self):  # -> Literal[False]:
        ...
    def estimate_kernel_num_bytes(self):  # -> int:

        ...
    def estimate_flops(self) -> int: ...
    def jit_lines(self):  # -> str:
        ...
    def gen_argdefs(self):  # -> str:
        ...
    def gen_defines(self): ...
    def def_kernel(self, *argnames):  # -> str:

        ...
    def size(self, name: str, index: int):  # -> str:

        ...
    def stride(self, name, index=...):  # -> str:

        ...
    def modification(
        self, subgraph_number: int, output_name: Optional[str], mask: Optional[str] = ..., **fixed_inputs
    ) -> str: ...
    def load_input(
        self,
        input_name: str,
        output_name: str,
        indices: Union[list[Any], tuple[Any]],
        mask: Optional[str] = ...,
        other: Optional[Union[float, int]] = ...,
        indent_width: int = ...,
    ):  # -> str:

        ...
    def store_output(
        self,
        indices: Union[list[Any], tuple[Any]],
        val: str,
        mask: Optional[str] = ...,
        indent_width: int = ...,
        val_shape: Optional[list[str]] = ...,
    ):  # -> str:

        ...
    def render(self, template, kwargs, record_input_dependent_tracked_event=...):  # -> PartialRender:
        ...
    def make_load(self, name, indices, mask):  # -> str:

        ...
    def indexing(
        self,
        index: sympy.Expr,
        *,
        dense_indexing=...,
        copy_shape=...,
        override_mask=...,
        block_ptr=...,
        tma_compatibility_checker: Optional[TMACompatibilityChecker] = ...,
    ):  # -> BlockDescriptorOptions | IndexingOptions:

        ...
    def codegen_range_tree(self):  # -> None:
        ...
    def additional_call_args_and_types(
        self,
    ):  # -> tuple[tuple[Any, Any, Any], map[type]] | tuple[Any, map[type]] | tuple[tuple[()], tuple[()]]:
        ...
    def call_kernel(self, name: str, node: Optional[ir.IRNode] = ...):  # -> None:
        ...
    def kernel_benchmark_extra_args(self) -> list[str]: ...

class GenerateAndLoadResult(NamedTuple):
    mod: ModuleType
    extra: str
    input_call_args: tuple[str, ...]
    prologue_supported_inputs: OrderedSet[str]
    kernel_args_sizevars_keys: tuple[sympy.Expr]
    kernel_options: dict[str, Any]

class GeneratedCodeCacheEntry(NamedTuple):
    code: str
    extra: str
    events: list[Any]

class GeneratedCodeCache:
    def __init__(self, *args, **kwargs) -> None: ...
    def cache_clear(self) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def make_key(
        self,
        input_nodes: tuple[ir.IRNode],
        num_stages: int,
        num_warps: int,
        call_sizes: Sequence[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Optional[Callable[..., Any]],
        epilogue_fn_hash: Optional[str],
        subgraphs: Optional[list[ir.Buffer]],
        workspace_arg: Optional[WorkspaceArg],
        layout: ir.Layout,
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        kwargs: dict[str, Any],
        hint_override: Optional[int] = ...,
    ) -> Optional[str]: ...
    def get_entry(self, cache_key: Optional[str]) -> Optional[GeneratedCodeCacheEntry]: ...
    def put_entry(self, cache_key: Optional[str], code: str, extra: str, events: list[Any]) -> None: ...

class TritonTemplate(KernelTemplate):
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
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> Optional[NotImplementedError]: ...
    def generate_and_load(
        self,
        input_nodes: tuple[ir.IRNode],
        num_stages: int,
        num_warps: int,
        call_sizes: Sequence[sympy.core.symbol.Symbol],
        prefix_args: int,
        suffix_args: int,
        epilogue_fn: Optional[Callable[..., Any]],
        epilogue_fn_hash: Optional[str],
        subgraphs: Optional[list[ir.Buffer]],
        workspace_arg: Optional[WorkspaceArg],
        num_consumer_groups: int,
        num_buffers_warp_spec: int,
        layout: ir.Layout,
        kwargs: dict[str, Any],
        generate_with_caching,
        hint_override: Optional[int] = ...,
    ) -> Optional[GenerateAndLoadResult]: ...
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
        epilogue_fn: Optional[Callable[..., Any]] = ...,
        epilogue_fn_hash: Optional[str] = ...,
        subgraphs: Optional[list[ir.Buffer]] = ...,
        mutated_inputs: Optional[list[ir.IRNode]] = ...,
        call_sizes: Optional[Sequence[sympy.core.symbol.Symbol]] = ...,
        workspace_arg: Optional[WorkspaceArg] = ...,
        generate_with_caching=...,
        hint_override: Optional[int] = ...,
        **kwargs,
    ):  # -> TritonTemplateCaller | None:

        ...

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
    def to_callable(self):  # -> Any:
        ...
    def call_name(self):  # -> str:
        ...
    @functools.cache
    def hash_key(self):  # -> str:
        ...
    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=..., **kwargs):  # -> ExternKernelCaller:
        ...
    @property
    def uid(self) -> str: ...
    def choice_or_none(self, **kwargs: Any) -> Optional[ChoiceCaller]: ...
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> Optional[NotImplementedError]: ...

class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def __init__(
        self,
        name,
        input_nodes,
        layout,
        make_kernel_render,
        description,
        bmreq,
        log_info: Optional[dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]] = ...,
        mutated_inputs=...,
        workspace_arg: Optional[WorkspaceArg] = ...,
        allowed_prologue_inps: Optional[OrderedSet[str]] = ...,
        hint_override: Optional[int] = ...,
    ) -> None: ...
    def benchmark(self, *args, out):  # -> float:
        ...
    def precompile(self):  # -> None:
        ...
    def call_name(self):  # -> str:
        ...
    def hash_key(self):  # -> str:
        ...
    def output_node(self):  # -> TensorBox | ShapeAsConstantBuffer:
        ...
    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]: ...
    def get_make_kernel_render(self): ...
    def autoheuristic_id(self):  # -> str:
        ...

class ExternKernelCaller(ChoiceCaller):
    def __init__(self, choice: ExternKernelChoice, input_nodes, layout, kwargs=..., *, has_out_variant=...) -> None: ...
    def benchmark(self, *args, out):  # -> float:
        ...
    def to_callable(self):  # -> partial[Any] | Any:
        ...
    def hash_key(self):  # -> str:
        ...
    def output_node(self):  # -> TensorBox | ShapeAsConstantBuffer:
        ...
    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]: ...
    def autoheuristic_id(self):  # -> str:
        ...

@functools.cache
def get_mm_log_filename() -> Optional[str]: ...
def append_to_log(filename, data):  # -> None:
    ...

class DataProcessorChoiceCallerWrapper:
    def __init__(self, wrapped, preprocessor, postprocessor) -> None: ...
    def __getattr__(self, name):  # -> Any:
        ...
    def benchmark(self, *args, out) -> float: ...
    def output_node(self) -> ir.TensorBox: ...

class DataProcessorTemplateWrapper:
    def __init__(self, wrapped_template_cls, preprocessor, postprocessor, **kwargs) -> None: ...
    def __getattr__(self, name):  # -> Any:
        ...
    def maybe_append_choice(self, choices, **kwargs): ...
    def generate(self, **kwargs):  # -> DataProcessorChoiceCallerWrapper:
        ...

class ErrorFromChoice(RuntimeError):
    def __init__(self, msg, choice: ChoiceCaller, inputs_str) -> None: ...

class NoValidChoicesError(RuntimeError): ...

@functools.cache
def get_num_workers() -> int: ...
def create_inputs_key(input_nodes) -> str: ...
def create_precompile_key(name: str, inputs_key: str, choices: list[ChoiceCaller]) -> str: ...

FeedbackFunction: TypeAlias = Callable[
    [dict[ChoiceCaller, float], str, list[Any], list[ChoiceCaller], Callable[[], dict[ChoiceCaller, float]]],
    None,
]
PreprocessingFunction: TypeAlias = Callable[[list[ChoiceCaller]], list[ChoiceCaller]]

def filter_choices_by_name_regex(choices: list[ChoiceCaller]) -> list[ChoiceCaller]: ...
def filter_choices_by_desc_regex(choices: list[ChoiceCaller]) -> list[ChoiceCaller]: ...

class AlgorithmSelectorCache(PersistentCache):
    def __init__(self, *args, **kwargs) -> None: ...
    def cache_clear(self) -> None: ...
    def __call__(
        self,
        name,
        choices: list[ChoiceCaller],
        input_nodes,
        layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]] = ...,
        precompilation_timeout_seconds: int = ...,
        return_multi_template=...,
        best_config_future=...,
    ):  # -> TensorBox | ShapeAsConstantBuffer:
        ...
    def make_precompile_fn(
        self, choices, name: str, inputs_key: str, precompilation_timeout_seconds: Optional[int] = ...
    ) -> Callable[[], None]: ...
    @classmethod
    def get_inputs(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = ...,
    ) -> AutotuneArgs: ...
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
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = ...,
    ) -> dict[ChoiceCaller, float]: ...
    @classmethod
    def benchmark_in_sub_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = ...,
    ):  # -> dict[ChoiceCaller, float]:
        ...
    @classmethod
    def make_benchmark_fn(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
        hint_override: Optional[int] = ...,
    ):  # -> partial[dict[ChoiceCaller, float]]:
        ...
    @staticmethod
    def prescreen_choices(
        choices: list[ChoiceCaller], name: str, inputs_key: str, prescreen_cache: dict[str, OrderedSet[str]]
    ) -> list[ChoiceCaller]: ...
    @staticmethod
    def prune_choices_postscreen(
        choices: list[ChoiceCaller],
        candidate_timings: dict[ChoiceCaller, float],
        name: str,
        inputs_key: str,
        prescreen_cache: dict[str, OrderedSet[str]],
    ) -> list[ChoiceCaller]: ...
    @staticmethod
    def log_results(
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict[ChoiceCaller, float],
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: Optional[float] = ...,
        hint_override: Optional[int] = ...,
    ):  # -> None:
        ...
    @staticmethod
    def benchmark_example_value(node, hint_override: Optional[int] = ...):  # -> Tensor:

        ...
    @staticmethod
    def generate_example_value(size, stride, device, dtype, extra_size, allocation_size=...):  # -> Tensor:
        ...
    @staticmethod
    def key_of(node):  # -> tuple[Any, str, *tuple[int, ...]]:

        ...
    def add_feedback_saver(self, fn: FeedbackFunction):  # -> None:
        ...
    def clear_feedback_savers(self):  # -> None:
        ...
    def add_preprocessing_fn(self, fn: PreprocessingFunction):  # -> None:
        ...
    def clear_preprocessing_fns(self, clear_defaults: bool = ...):  # -> None:

        ...

_ALGORITHM_SELECTOR_CACHE: Optional[AlgorithmSelectorCache] = ...

def get_algorithm_selector_cache() -> AlgorithmSelectorCache: ...
def autotune_select_algorithm(*args, **kwargs):  # -> TensorBox | ShapeAsConstantBuffer:
    ...
def add_feedback_saver(fn: FeedbackFunction):  # -> None:
    ...
def clear_feedback_savers():  # -> None:

    ...
def add_preprocessing_fn(fn: PreprocessingFunction):  # -> None:

    ...
def clear_preprocessing_fns(clear_defaults: bool = ...):  # -> None:

    ...
def realize_inputs(*args):  # -> Any | list[Any | list[Any | list[Any]]]:
    ...

class SymbolicGridFn:
    def __init__(self, fn: Callable[..., tuple[Any, Any, Any]]) -> None: ...
    def __call__(self, *args, **kwargs) -> tuple[int, int, int]: ...
    def sympy_call(self, *args, **kwargs):  # -> tuple[Any, Any, Any]:
        ...
