import contextlib
import torch
import torch.nn as nn
from typing import Callable, Optional
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.utils import BoxedBool
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from ._aot_autograd.descriptors import AOTInput
from ._aot_autograd.schemas import (
    AOTConfig,
    AOTDispatchCompiler,
    AOTState,
    FakifiedFlatArgs,
    GraphSignature,
    JointWithDescriptors,
)

static_inputs_log = ...
zip = ...
AOT_COUNTER = ...
aot_autograd_decompositions = ...

def create_aot_state(
    stack: contextlib.ExitStack,
    flat_fn,
    fake_flat_args: FakifiedFlatArgs,
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: Optional[ShapeEnv],
) -> AOTState: ...
def aot_function(
    fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = ...,
    partition_fn: Callable = ...,
    decompositions: Optional[dict] = ...,
    num_params_buffers: int = ...,
    keep_inference_input_mutations: bool = ...,
    inference_compiler: Optional[Callable] = ...,
    *,
    dynamic=...,
    enable_log=...,
) -> Callable: ...
def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module: ...
def prepare_aot_module_simplified(
    mod: nn.Module,
    args,
    kwargs,
    fw_compiler: Optional[AOTDispatchCompiler],
    bw_compiler: Optional[AOTDispatchCompiler],
    partition_fn: Callable,
    decompositions: dict,
    keep_inference_input_mutations,
    inference_compiler: Optional[AOTDispatchCompiler],
    boxed_forward_device_index: BoxedDeviceIndex,
    ignore_shape_env: bool,
    flatten: bool,
    *,
    force_non_lazy_backward_lowering: bool = ...,
):  # -> tuple[Callable[..., Any] | Callable[..., Any | tuple[Any, ...] | list[Any]], list[Tensor | Parameter], list[str], list[str], FakifiedFlatArgs, list[Any], AOTConfig, FakeTensorMode, ShapeEnv | None, TreeSpec | None, PytreeThunk | None]:
    ...
def aot_module_simplified(
    mod: nn.Module,
    args,
    fw_compiler: AOTDispatchCompiler,
    bw_compiler: Optional[AOTDispatchCompiler] = ...,
    partition_fn: Callable = ...,
    decompositions: Optional[dict] = ...,
    keep_inference_input_mutations=...,
    inference_compiler: Optional[AOTDispatchCompiler] = ...,
    cudagraphs: Optional[BoxedBool] = ...,
    boxed_forward_device_index: Optional[BoxedDeviceIndex] = ...,
    ignore_shape_env: bool = ...,
) -> nn.Module: ...
def boxed_nop_preserve_node_meta(fx_g, example_inputs):  # -> Callable[..., Any]:
    ...
def aot_export_joint_with_descriptors(
    stack: contextlib.ExitStack,
    mod: nn.Module,
    args,
    kwargs=...,
    *,
    decompositions: Optional[dict] = ...,
    keep_inference_input_mutations=...,
    ignore_shape_env=...,
    fw_compiler: Optional[AOTDispatchCompiler] = ...,
    bw_compiler: Optional[AOTDispatchCompiler] = ...,
) -> JointWithDescriptors: ...
def aot_compile_joint_with_descriptors(jd: JointWithDescriptors) -> callable: ...
def aot_export_module(
    mod: nn.Module,
    args,
    *,
    decompositions: Optional[dict] = ...,
    trace_joint: bool,
    output_loss_index: Optional[int] = ...,
    pre_dispatch: bool = ...,
    dynamic_shapes: Optional[bool] = ...,
    kwargs=...,
) -> tuple[torch.fx.GraphModule, GraphSignature]: ...
def aot_export_joint_simple(
    func: Callable, args, *, trace_joint: bool, num_params_buffers: int = ..., decompositions: Optional[dict] = ...
) -> torch.fx.GraphModule: ...

compiled_function = ...
compiled_module = ...
