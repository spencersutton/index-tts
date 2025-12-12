import dataclasses
import torch
from typing import Any, Callable, Optional, TYPE_CHECKING, Union
from typing_extensions import TypeAlias
from torch._inductor.cudagraph_utils import BoxedDeviceIndex, CudagraphCachedInfo
from torch._inductor.utils import BoxedBool, GraphPartitionMap, InputType
from torch.utils._ordered_set import OrderedSet
from collections import Counter
from collections.abc import Sequence
from torch._inductor import metrics
from torch._inductor.graph import GraphLowering
from torch._library.fake_class_registry import FakeScriptObject
from torch.export.pt2_archive._package_weights import Weights
from .compile_fx import _CompileFxKwargs
from .triton_bundler import TritonBundle

"""
This provides an abstract class which parametrizes over an "output code" concept
for Inductor.  Intuitively, this represents the compiled callable which Inductor
produces which you can call to get optimized code.  However, this callable
has some other capabilities:

- It is serializable, so you can save/load this product from disk without
  having to do compilation again.

- (When using remote cache) it is addressable, so you can save just a key
  which you can use to load this product from remote cache later.

This class is abstract because we have several different implementations of
serialized format:

- Python wrapper (the default)

- AOTInductor (this produces ABI stable binaries which work across PyTorch
  versions)

"""
if TYPE_CHECKING: ...
log = ...

@dataclasses.dataclass
class OutputCode:
    _fx_graph_cache_key: Optional[str] = ...
    _fx_graph_cache_debug_lines: Optional[list[str]] = ...
    _time_taken_ns: Optional[int] = ...
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...

_StrideExprStr: TypeAlias = str

def get_expanded_dims(t: torch.Tensor) -> list[int]: ...
def index_expanded_dims(t: torch.Tensor, expanded_dims: list[int]) -> torch.Tensor: ...
def complex_memory_overlap(t: torch.Tensor) -> bool: ...
def maybe_handle_backward_generation(
    compiled_graph: CompiledFxGraph, boxed_forward_device_index: Optional[BoxedDeviceIndex]
) -> None: ...
def prepare_cudagraph_post_compile(
    compiled_graph: CompiledFxGraph,
    example_inputs: Sequence[InputType],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None: ...
def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None: ...
def cudagraph_partition_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None: ...
def maybe_realign_inputs(
    ran_cudagraphs: BoxedBool,
    compiled_graph: CompiledFxGraph,
    inputs_to_check: Sequence[int],
    mutated_inputs_idxs: OrderedSet[int],
) -> None: ...

class CompiledFxGraphConstants:
    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor]: ...

class CompiledFxGraphConstantsWithGm(CompiledFxGraphConstants):
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...
    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor]: ...

@dataclasses.dataclass
class CompiledFxGraph(OutputCode):
    current_callable: Optional[Callable[..., Any]]
    recursively_apply_fns: Optional[Callable[..., Any]]
    compiled_fn_runner: Optional[Any]
    cache_key: str
    source_code: str = ...
    runnable_graph_str: str = ...
    inductor_post_grad_graph_str: str = ...
    cache_linemap: Optional[list[tuple[int, str]]]
    device_types: OrderedSet[str]
    device_idxs: OrderedSet[int]
    mutated_inputs: OrderedSet[str]
    mutated_input_idxs: OrderedSet[int]
    constants: Optional[dict[str, torch.Tensor]]
    frozen_param_names: dict[str, str]
    torchbind_constants: dict[str, torch._C.ScriptObject | FakeScriptObject]
    output_strides: Optional[list[Optional[tuple[_StrideExprStr, ...]]]]
    disabled_cudagraphs_reason: Optional[str]
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    guards_expr: Optional[str]
    inductor_provenance_mapping_str: Optional[str]
    inductor_provenance_stack_traces_str: Optional[str]
    cudagraph_info: Optional[CudagraphCachedInfo]
    partition_maps: Optional[list[GraphPartitionMap]]
    fx_kwargs: _CompileFxKwargs
    inputs_to_check: Sequence[int]
    _boxed_call: Optional[bool] = ...
    _triton_bundle: Optional[TritonBundle] = ...
    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        gm: torch.fx.GraphModule,
        output_strides: list[Optional[tuple[_StrideExprStr, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
        cudagraphs: BoxedBool,
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        runnable_graph_str: str,
        inductor_post_grad_graph_str: str,
        compiled_fn_runner: Optional[Any] = ...,
        inductor_provenance_mapping_str: Optional[str] = ...,
        inductor_provenance_stack_traces_str: Optional[str] = ...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...
    def prepare_for_serialization(self) -> None: ...
    def write_to_disk(self) -> str: ...
    def after_deserialization(self, constants: CompiledFxGraphConstants) -> str: ...

@dataclasses.dataclass
class CompiledAOTI(OutputCode):
    filename: Union[str, list[Union[str, Weights]], torch.fx.GraphModule]
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...

@dataclasses.dataclass
class MockFXGraphCacheOutput(OutputCode):
    gm: Any = ...
    def __post_init__(self) -> None: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...
