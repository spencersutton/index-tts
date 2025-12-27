"""Utils for caching the outputs of AOTAutograd"""

import contextlib
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, override

import torch
from torch._dynamo.precompile_context import PrecompileCacheArtifact
from torch._inductor.codecache import FxGraphCachePickler, FxGraphHashDetails, GuardedCache
from torch._inductor.compile_fx import _CompileFxKwargs
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.output_code import CompiledFxGraph, OutputCode
from torch._inductor.remote_cache import JsonDataTy, RemoteCache
from torch._inductor.utils import BoxedBool
from torch.compiler._cache import CacheArtifact, CacheArtifactFactory
from torch.fx.node import Node

from .runtime_wrappers import CompilerWrapper, SubclassMeta
from .schemas import AOTAutogradCacheInfo, AOTConfig, ViewAndMutationMeta

log = ...

class BypassAOTAutogradCache(Exception): ...
class FXGraphCacheMiss(BypassAOTAutogradCache): ...

def should_use_remote_autograd_cache(): ...
def should_use_local_autograd_cache(): ...
def should_bundle_autograd_cache(): ...
def check_node_safe(node: Node):
    """
    Checks that the node only uses supported operators. We are starting with very
    conservative cacheability constraints, and incrementally adding more support as we expand.

    [Note: AOTAutograd Cacheability checks]
    - Our cache key is computed from the FX graph produced by Dynamo and the input example values
    - A node is "safe" if the same cache key results in a compiled artifact that has the same behavior
        (i.e, the set of inputs that go into our cache key is sufficient to distinguish its behavior)

    To accomplish this safety check, we consider the following functions to be safe:
        - Public functions under modules torch, torch.functional, and torch.nn.functional: these are
        allowed in the graph by dynamo, so we can assume they are safe to cache.
        - method calls on base tensor types
        - Any call_module that dynamo deemed safe to allow AOTAutograd to trace
        - Non callable nodes, such as placeholder, output, get_attr

    The test suite test_aot_autograd_cache.py::AOTAutogradCachePicklerTests tries its best to fully cover/specify this behavior.
    """

def check_cacheable(gm: torch.fx.GraphModule):
    """Checks that the graph module only uses supported operators"""

class AOTAutogradCacheDetails(FxGraphHashDetails):
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """
    def get_triton_source_codes_from_gm(self, gm: torch.fx.GraphModule): ...
    def __init__(
        self, gm: torch.fx.GraphModule, example_inputs, aot_config: AOTConfig, fx_config: _CompileFxKwargs
    ) -> None: ...

class AOTAutogradCachePickler(FxGraphCachePickler):
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...

@contextlib.contextmanager
def normalize_placeholder_names(gm: torch.fx.GraphModule):
    """
    Context manager that normalizes the placeholder names in the graph module.
    This is used while generating a cache key for AOTAutogradCache, so that two graphs
    that are isomorphic when normalizing names can hit the same cache entry.
    This is safe because nothing underneath AOTAutograd uses the node names on the
    original dynamo graph: AOTAutograd re-traces with its own nodes, and guards are
    in terms of original sources rather than placeholder names.
    """

def autograd_cache_key(
    gm: torch.fx.GraphModule, example_inputs, config: AOTConfig, fx_config: _CompileFxKwargs
) -> tuple[str, list[str]]:
    """Generate a unique hash of the FX graph for caching."""

TOut = TypeVar("TOut", bound=OutputCode)

class InductorOutput[TOut: OutputCode](ABC):
    """Class representing a single inductor output"""
    @abstractmethod
    def pre_save(self) -> None: ...
    @abstractmethod
    def load(self, example_inputs) -> TOut: ...
    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...

@dataclass
class CompiledFxGraphLoadable(InductorOutput[CompiledFxGraph]):
    """
    A full compiled fx graph that doesn't need to lookup the FxGraphCache
    to run
    """

    result: CompiledFxGraph
    def pre_save(self) -> None: ...
    def load(self, example_inputs) -> CompiledFxGraph: ...
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph: ...

@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    """FxGraphCacheLoadable(fx_graph_cache_info: 'tuple[str, list[str]]', fx_graph_guard_expr: 'Optional[str]')"""

    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: str | None
    def pre_save(self): ...
    def load(self, example_inputs) -> CompiledFxGraph: ...
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph:
        """Called after FXGraphCacheLoadable.load, mutates fx_config"""

@dataclass
class CompiledForward(FxGraphCacheLoadable):
    """Cacheable entry for a forward function"""

@dataclass
class GenericCompiledBackward(InductorOutput[TOut]):
    """GenericCompiledBackward(backward_state_indices: 'list[int]', num_symints_saved_for_bw_: 'int')"""

    backward_state_indices: list[int]
    num_symints_saved_for_bw_: int

@dataclass
class CompiledBackward(GenericCompiledBackward[CompiledFxGraph], FxGraphCacheLoadable):
    """Cacheable entry for a forward function"""
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph: ...

class BundledCompiledForward(CompiledFxGraphLoadable): ...

@dataclass
class BundledCompiledBackward(GenericCompiledBackward[CompiledFxGraph], CompiledFxGraphLoadable):
    """BundledCompiledBackward(result: 'CompiledFxGraph', backward_state_indices: 'list[int]', num_symints_saved_for_bw_: 'int')"""
    def post_compile(self, result: CompiledFxGraph, fx_config: _CompileFxKwargs) -> CompiledFxGraph: ...

@dataclass
class SerializedGraphModule:
    """SerializedGraphModule(gm: 'torch.fx.GraphModule')"""

    fn: Callable[[dict[Any, Any], str], torch.nn.Module]
    args: tuple[Any, ...]
    def __init__(self, gm: torch.fx.GraphModule) -> None: ...
    def deserialize(self) -> torch.fx.GraphModule: ...

def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule: ...

TForward = TypeVar("TForward", bound=InductorOutput)
TBackward = TypeVar("TBackward", bound=GenericCompiledBackward)

@dataclass
class GenericAOTAutogradCacheEntry[TForward: InductorOutput, TBackward: GenericCompiledBackward]:
    """
    A single entry into the cache, genericized by Forward and Backward types.

    A TForward is always an InductorOutput of some sort, which represents the
    forward graph of the compile.
    A TBackward is an InductorOutput + metadata about the backward, useful for specific
    backward-only wrappers. This type is encapsulated by GenericCompiledBackward.

    Each AOTAutogradCacheEntry is essentially parameterized by 1. the method of loading
    from the cache (either Bundled or UnBundled), and 2. The type of the output. For now,
    the only type of output we support is Python Wrapper output, i.e. OutputCode.CompiledFxGraph,
    but the same technique works for C++ wrapper code; we'd just add an extra InductorOutput type.
    """

    compiled_fw: TForward
    compiled_bw: TBackward | None
    aot_joint_graph_str: str | None
    aot_forward_graph_str: str | None
    aot_backward_graph_str: str | None
    runtime_metadata: ViewAndMutationMeta
    dispatch_wrappers: list[CompilerWrapper]
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None
    indices_of_inps_to_detach: list[int]
    forward_time_taken_ns: int
    backward_time_taken_ns: int
    sanitized_aot_config: AOTConfig
    guards_expr: str | None
    serialized_bw_module: SerializedGraphModule | None
    def pre_save(self):
        """Perform any preparations to make the cache entry ready for serialization."""
    def wrap_post_compile(
        self, args: list[torch.Tensor], aot_config: AOTConfig, fx_config: _CompileFxKwargs
    ) -> Callable:
        """
        This function takes a cache entry and carefully reconstructs the original callable
        that AOTAutograd returned the first time it was run. It does this by running the various
        post compile steps that AOTAutograd runs on its compiled artifact after running the fw/bw compilers.

        In the inference path, this consists of the Subclass, FunctionalzedRngRuntime, and RuntimeWrappers.
        In the autograd path, this consists of AOTAutogradDispatch.post_compile.

        The steps here should match exactly the steps that are run in aot_dispatch_base and aot_dispatch_autograd.

        Notably absent from the cached path are:
        - DebugAssertWrapper
        - FakifiedOutWrapper

        Which we'll handle separately later on, if necessary.
        """

class AOTAutogradCacheEntry(GenericAOTAutogradCacheEntry[CompiledForward, CompiledBackward]):
    """
    Regular AOTAutogradCacheEntry: saves the forward/backward FxGraphCache keys
    and looks them up in FxGraphCache on load
    """

class BundledAOTAutogradCacheEntry(GenericAOTAutogradCacheEntry[BundledCompiledForward, BundledCompiledBackward]):
    """
    AOTAutogradCacheEntry where we save the entire CompiledFxGraph instead
    of relying on cache keys from FxGraphCache
    """

@contextlib.contextmanager
def sanitize_gm_for_cache(gm: torch.fx.GraphModule):
    """
    Clears a few fields in a dynamo supplied Graph Module that are not stable between graph inputs, but don't
    affect inductor or aotdispatch correctness.

    These fields **can** be used by code calling into aotdispatch (namely, dynamo), so we can't null them out completely.

    To ensure that these fields are not accessed by inductor or aotdispatch, we clear them during AOTAutogradCache.load,
    and then put them back before returning. This way, we generate a cache key based off of a canonical graph
    without these fields, and also guarantee they aren't used to affect the cache's output.
    """

@CacheArtifactFactory.register
class AOTAutogradCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self): ...
    @override
    @staticmethod
    def type(): ...

@CacheArtifactFactory.register
class BundledAOTAutogradCacheArtifact(PrecompileCacheArtifact[Callable]):
    @override
    @staticmethod
    def type(): ...
    @override
    def after_deserialization(self) -> Callable: ...

class AOTAutogradCache(GuardedCache[GenericAOTAutogradCacheEntry]):
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradCacheEntry handles the wrapping/unwrapping logic.

    Cache Inputs (AOTAutogradCacheDetails)
    - AOTAutogradCache takes in the following inputs, which are analogous to inputs given
        to AOTAutograd by dynamo:
        - A fx graph module generated by dynamo
        - A list of args, which consists of:
            - Symint inputs to the graph, generated by dynamo
            - The **real tensor** inputs, which inductor uses for cudagraphs
            - Notably, the real tensor inputs don't have symints in their metadata.
        AOTAutograd then retraces those real tensor arguments into FakeTensors later during execution.
        - A set of global configurations that affect AOTAutograd or Inductor behavior.

    It then generates a cache key given these values. Notably, this means AOTAutogradCache currently
    specializes on the sizes and strides of the real tensor inputs when dynamic shapes are turned on.
    In a later PR, we'll likely generate the cache key based on the FakeTensors AOTAutograd generates
    based on the real tensor inputs, which can contain symints.

    # Cache Outputs (AOTAutogradCacheEntry)
    - AOTAutogradCache caches the following values:
        - The compiled forward and backward functions from inductor, via keys to the FXGraphCache
        - Metadata to reconstruct the AOTModule from the compiled inductor artifacts
        - See AOTAutogradCacheEntry for more info

    [Note: Caching guards generated by AOTAutograd and Inductor]
    AOTAutograd and inductor both can introduce new guards to the shape environment. FXGraphCache saves guards with each
    compiled graph inductor generates. On a cache hit, AOTAutograd reloads the compiled forward and backward functions
    from FXGraphCache, giving it new symint arguments from the input args.
    FXGraphCache uses those symints and its saved guards to repopulate the ShapeEnv with guards.
    **No new guards are generated into the shape env after inductor finishes compiling**, so the guards
    saved by inductor are sufficient for correctness for both AOTAutograd and Inductor's caches.
    """
    @staticmethod
    def clear():
        """Clear the cache"""
    @staticmethod
    def try_load(
        mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
        args,
        aot_config: AOTConfig,
        cudagraphs: BoxedBool,
        boxed_forward_device_index: BoxedDeviceIndex | None,
        local: bool,
        remote: bool,
    ) -> Callable | None:
        """Load a result from the cache, and reconstruct a runtime wrapper around the object"""
    @classmethod
    def generate_guards_expression(cls: type[AOTAutogradCache], cache_info: AOTAutogradCacheInfo) -> str | None: ...
    @staticmethod
    def evaluate_guards(guard_expr: str, hints: list[int] | list[torch.SymInt]): ...
    @staticmethod
    def save(key: str, entry: GenericAOTAutogradCacheEntry, remote: bool):
        """Save a single entry into the cache."""
    @staticmethod
    @functools.cache
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
        """Attempts to load the remote cache, returns None on error."""
    @staticmethod
    def make_entry(
        compiled_fw_func: CompiledFxGraph,
        compiled_bw_func: CompiledFxGraph | None,
        aot_joint_graph_str: str | None,
        aot_forward_graph_str: str | None,
        aot_backward_graph_str: str | None,
        runtime_metadata: ViewAndMutationMeta,
        dispatch_wrappers: list[CompilerWrapper],
        maybe_subclass_meta: SubclassMeta | None,
        num_fw_outs_saved_for_bw: int | None,
        indices_of_inps_to_detach: list[int],
        forward_time_taken_ns: int,
        backward_time_taken_ns: int,
        sanitized_aot_config: AOTConfig,
        guards_expr: str | None,
        backward_state_indices: list[int] | None,
        num_symints_saved_for_bw: int | None,
        serialized_bw_module: SerializedGraphModule | None,
    ) -> GenericAOTAutogradCacheEntry: ...
