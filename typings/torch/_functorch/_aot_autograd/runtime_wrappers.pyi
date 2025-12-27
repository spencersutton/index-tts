"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx
from torch._guards import CompileContext, TracingContext

from .descriptors import AOTInput
from .schemas import (
    AOTConfig,
    CompilerWrapper,
    FxValue,
    InductorWrapper,
    InputAliasInfo,
    MemoryFormatMeta,
    PlainTensorMeta,
    SubclassCreationMeta,
    SubclassMeta,
    TraceFn,
    ViewAndMutationMeta,
)

zip = ...

@dataclass
class RuntimeWrapper(CompilerWrapper):
    """RuntimeWrapper(indices_of_inps_to_detach: list[int], trace_joint: bool, disable_amp: bool)"""

    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

class NoopAliasHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class AliasOfInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class IsInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class AliasOfIntermediateHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

_HANDLER_MAP = ...

def make_output_handler(info, runtime_metadata, trace_joint): ...
def maybe_mark_dynamic_helper(t: torch.Tensor, dims: set[int]): ...

@dataclass
class FunctionalizedRngRuntimeWrapper(InductorWrapper):
    """FunctionalizedRngRuntimeWrapper(return_new_outs: bool = True)"""

    return_new_outs: bool = ...
    def pre_compile(self, flat_fn: torch.fx.GraphModule, flat_args, aot_config, *, fw_metadata) -> None: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class FakifiedOutWrapper(InductorWrapper):
    """FakifiedOutWrapper(out_metas: list[torch.Tensor] = <factory>, fwd_output_strides: Optional[list[Optional[list[int]]]] = None, needs_post_compile: bool = True)"""

    out_metas: list[torch.Tensor] = ...
    fwd_output_strides: list[list[int] | None] | None = ...
    needs_post_compile: bool = ...
    def pre_compile(self, fw_module: fx.GraphModule, flat_args, aot_config, *, fw_metadata) -> None: ...
    def set_fwd_output_strides(self, fwd_output_strides): ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    """AOTDispatchSubclassWrapper(trace_joint: bool, fw_only: Optional[Callable], maybe_subclass_meta: Optional[torch._functorch._aot_autograd.schemas.SubclassMeta], num_fw_outs_saved_for_bw: Optional[int])"""

    trace_joint: bool
    fw_only: Callable | None
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None
    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ): ...
    def post_compile(self, compiled_fn, _aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class EffectTokensWrapper(CompilerWrapper):
    """EffectTokensWrapper()"""
    def post_compile(self, compiled_fn, _aot_config, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    """AOTDedupeWrapper(keep_arg_mask: list[bool] = <factory>, add_dupe_map: list[int] = <factory>, old_input_metadata: list[torch._functorch._aot_autograd.schemas.InputAliasInfo] = <factory>, needs_post_compile: bool = True)"""

    keep_arg_mask: list[bool] = ...
    add_dupe_map: list[int] = ...
    old_input_metadata: list[InputAliasInfo] = ...
    needs_post_compile: bool = ...
    def remove_dupe_args(self, args): ...
    def add_dupe_args(self, args): ...
    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    """AOTSyntheticBaseWrapper(trace_joint: bool, needs_post_compile: bool = True, aliased_arg_idx_with_metadata_mutations: list[int] = <factory>)"""

    trace_joint: bool
    needs_post_compile: bool = ...
    aliased_arg_idx_with_metadata_mutations: list[int] = ...
    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[Callable, list[FxValue], list[AOTInput], ViewAndMutationMeta]: ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

def merge_view_inputs(
    aot_config: AOTConfig,
    fwd_inputs: list[Any],
    fwd_inputs_descs: list[AOTInput] | None,
    mutated_input_info: list[InputAliasInfo],
    *,
    is_inference: bool,
) -> tuple[list[Any], list[AOTInput], list[int | tuple[int, torch.Tensor]] | None]: ...

@dataclass
class AutogradLazyBackwardCompileInfo:
    """AutogradLazyBackwardCompileInfo(bw_module: Callable, placeholder_list: list[typing.Any], saved_context: Optional[torch._guards.TracingContext], saved_compile_context: Optional[torch._guards.CompileContext])"""

    bw_module: Callable
    placeholder_list: list[Any]
    saved_context: TracingContext | None
    saved_compile_context: CompileContext | None

@dataclass
class CachedAutogradLazyBackwardCompileInfo:
    """CachedAutogradLazyBackwardCompileInfo(bw_module_fn: Callable)"""

    bw_module_fn: Callable

def initialize_rng_states(
    num_rng: int, graphsafe_idx: int, fwd_rng_states: list[torch.Generator], bwd_rng_states: list[torch.Generator]
):
    """
    Initialize the cudagraph safe rng states.

    Initialization of rng states should have a few properties:
    - the initialization for each rng state should be independent
    - the initialization should be deterministic
    - the initialization should be based off current rng state, so that independent graphs do not
    have equal rng behavior

    We defer initialization of rng states until runtime because compilation is wrapped
    with preserve_rng_states. Seed initialization should advance the rng states so consecutive compilations
    do not give equal randomness.
    """

def coerce_to_expected_memory_format(x: torch.Tensor, memory_format: MemoryFormatMeta): ...

class AOTDispatchAutograd:
    @staticmethod
    def process_runtime_tangent(x, meta: PlainTensorMeta | SubclassCreationMeta): ...
    @staticmethod
    def post_compile(
        compiled_fw_func,
        compiled_bw_func,
        maybe_subclass_meta: SubclassMeta | None,
        num_symints_saved_for_bw_: int,
        backward_state_indices: list[int],
        disable_amp: bool,
        indices_of_inps_to_detach: list[int],
        lazy_backward_info: AutogradLazyBackwardCompileInfo | CachedAutogradLazyBackwardCompileInfo | None,
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
        try_save_cache_entry: Callable | None,
    ) -> None: ...

@dataclass
class DebugAssertWrapper(CompilerWrapper):
    """DebugAssertWrapper(flat_requires_grad: list[typing.Optional[bool]] = <factory>)"""

    flat_requires_grad: list[bool | None] = ...
    def post_compile(self, compiled_fn, aot_config: AOTConfig, *, runtime_metadata: ViewAndMutationMeta): ...

def pre_compile(
    wrappers: list[CompilerWrapper],
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function and arguments.
    Mutates wrappers in place.
    """

def post_compile(
    wrappers: list[CompilerWrapper],
    compiled_fn: Callable,
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> tuple[Callable, ViewAndMutationMeta]:
    """Runs a sequence of wrappers on the given function. Should be called after pre_compile()"""

def make_runtime_safe(fw_metadata: ViewAndMutationMeta, maybe_subclass_meta: SubclassMeta | None):
    """
    Calls make_runtime_safe on all ViewAndMutationMetas.
    Modifies both arguments. Allows ViewAndMutationMetas to
    be safely cached in AOTAutogradCache.
    """
