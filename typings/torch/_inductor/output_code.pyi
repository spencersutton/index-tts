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

import dataclasses
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor import metrics
from torch._inductor.cudagraph_utils import BoxedDeviceIndex, CudagraphCachedInfo
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import BoxedBool, GraphPartitionMap, InputType
from torch._library.fake_class_registry import FakeScriptObject
from torch.export.pt2_archive._package_weights import Weights
from torch.utils._ordered_set import OrderedSet

from .compile_fx import _CompileFxKwargs
from .triton_bundler import TritonBundle

log = ...

@dataclasses.dataclass
class OutputCode:
    """OutputCode()"""

    _fx_graph_cache_key: str | None = ...
    _fx_graph_cache_debug_lines: list[str] | None = ...
    _time_taken_ns: int | None = ...
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...

type _StrideExprStr = str

def get_expanded_dims(t: torch.Tensor) -> list[int]: ...
def index_expanded_dims(t: torch.Tensor, expanded_dims: list[int]) -> torch.Tensor: ...
def complex_memory_overlap(t: torch.Tensor) -> bool: ...
def maybe_handle_backward_generation(
    compiled_graph: CompiledFxGraph, boxed_forward_device_index: BoxedDeviceIndex | None
) -> None: ...
def prepare_cudagraph_post_compile(
    compiled_graph: CompiledFxGraph,
    example_inputs: Sequence[InputType],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None: ...
def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    """
    Checks for any reasons not to run cudagraphs and then
    runs it on compiled_graph.
    Mutates the `compiled_graph.current_callable` and `cudagraphs`
    """

def cudagraph_partition_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    """
    Cudagraphify each partition functions, which first prepares the necessary
    metadata and then applies the cudagraphify function to each partition.

    Assuming all partition functions are cudagraphified and share the same order
    as `compiled_graph.partition_maps`. See [Note: Graph Partition Map for CUDAGraph].
    """

def maybe_realign_inputs(
    ran_cudagraphs: BoxedBool,
    compiled_graph: CompiledFxGraph,
    inputs_to_check: Sequence[int],
    mutated_inputs_idxs: OrderedSet[int],
) -> None:
    """
    Realigns input strides from inputs_to_check if
    we didn't end up running cudagraphs. Mutates
    `compiled_graph.current_callable` if cudagraphs
    was run. Otherwise, does nothing.
    """

class CompiledFxGraphConstants:
    """
    Wrapper class that unwraps constants from a compiled fx graph. This
    version of the class only supports directly grabbing the saved constants off of
    a CompiledFxGraph.

    With freezing, FxGraphCache doesn't store the constants of the input
    GraphModule it gets from AOTAutograd. Instead, it saves just the **names**
    of those constants, and grabs the constant values directly from the graph module
    passed in at runtime.

    Thing is, we don't always *have* the graph module available at runtime, hence
    the existence of this class and its CompiledFxGraphConstantsWithGm counterpart.

    To support freezing, FXGraphCache gets passed a CompiledFxGraphConstantsWithGm during
    post compile. Otherwise, CompiledFxGraphConstants supports the basic case of loading
    the value of constants directly off of the original saved object.
    """
    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor]: ...

class CompiledFxGraphConstantsWithGm(CompiledFxGraphConstants):
    """
    This version of CompiledFxGraphConstants, instead of grabbing constants
    directly saved on CompiledFxGraphs, will just grab their names. Then, it takes
    a second GraphModule to grab the corresponding constant values out of.

    This is necessary for supporting freezing in FxGraphCache.
    """
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...
    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor]: ...

@dataclasses.dataclass
class CompiledFxGraph(OutputCode):
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Callable[..., Any] | None
    recursively_apply_fns: Callable[..., Any] | None
    compiled_fn_runner: Any | None
    cache_key: str
    source_code: str = ...
    runnable_graph_str: str = ...
    inductor_post_grad_graph_str: str = ...
    cache_linemap: list[tuple[int, str]] | None
    device_types: OrderedSet[str]
    device_idxs: OrderedSet[int]
    mutated_inputs: OrderedSet[str]
    mutated_input_idxs: OrderedSet[int]
    constants: dict[str, torch.Tensor] | None
    frozen_param_names: dict[str, str]
    torchbind_constants: dict[str, torch._C.ScriptObject | FakeScriptObject]
    output_strides: list[tuple[_StrideExprStr, ...] | None] | None
    disabled_cudagraphs_reason: str | None
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    guards_expr: str | None
    inductor_provenance_mapping_str: str | None
    inductor_provenance_stack_traces_str: str | None
    cudagraph_info: CudagraphCachedInfo | None
    partition_maps: list[GraphPartitionMap] | None
    fx_kwargs: _CompileFxKwargs
    inputs_to_check: Sequence[int]
    _boxed_call: bool | None = ...
    _triton_bundle: TritonBundle | None = ...
    def __init__(
        self,
        current_callable: Callable[..., Any] | None,
        graph: GraphLowering,
        gm: torch.fx.GraphModule,
        output_strides: list[tuple[_StrideExprStr, ...] | None],
        disabled_cudagraphs_reason: str | None,
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
        cudagraphs: BoxedBool,
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        runnable_graph_str: str,
        inductor_post_grad_graph_str: str,
        compiled_fn_runner: Any | None = ...,
        inductor_provenance_mapping_str: str | None = ...,
        inductor_provenance_stack_traces_str: str | None = ...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None:
        """
        Run a set of post processing steps after loading from the cache. These involve:
         - Setting the tracing context output strides
         - Running cudagraphs if enabled
         - Realigning inputs

        This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
        The results of this function are *not* saved in the cache itself.
        """
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...
    def prepare_for_serialization(self) -> None: ...
    def write_to_disk(self) -> str: ...
    def after_deserialization(self, constants: CompiledFxGraphConstants) -> str: ...

@dataclasses.dataclass
class CompiledAOTI(OutputCode):
    """Class holding an AOTInductor compiled so."""

    filename: str | list[str | Weights] | torch.fx.GraphModule
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...

@dataclasses.dataclass
class MockFXGraphCacheOutput(OutputCode):
    """MockFXGraphCacheOutput(gm: 'Any' = None)"""

    gm: Any = ...
    def __post_init__(self) -> None: ...
    def post_compile(
        self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs
    ) -> None: ...
    def __call__(self, inputs: Sequence[Any]) -> Any: ...
    def set_triton_bundle(self, triton_bundle: Any) -> None: ...
