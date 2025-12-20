from typing import Any

import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses import FakeTensorMode
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode

Tensor = torch.Tensor
__all__ = ["trace_wrapped"]

@torch.library.custom_op("flex_lib::zeros_and_scatter", mutates_args=())
def zeros_and_scatter(shape: list[int], indices: list[Tensor], vals: Tensor) -> Tensor: ...
@zeros_and_scatter.register_fake
def _(shape: list[int], indices: list[Tensor], vals: Tensor) -> Tensor: ...
@zeros_and_scatter.register_vmap
def _(info, indims, shape, indices, value): ...

class ModIndex(torch.autograd.Function):
    generate_vmap_rule = ...
    @staticmethod
    def forward(x: Tensor, indices: list[Tensor]) -> Tensor: ...
    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None: ...
    @staticmethod
    def backward(ctx, gradOut): ...
    @classmethod
    @torch._export.wrappers.allow_in_pre_dispatch_graph
    def apply(cls, *args, **kwargs): ...

mod_index = ...

class TransformGetItemToIndex(TorchFunctionMode):
    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...

def trace_wrapped(*args: Any, **kwargs: Any) -> Any: ...

class TraceWrapped(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

_trace_wrapped_op = ...

@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(
    mode: ProxyTorchDispatchMode, *args: Any, bw_state: BackwardState | None = ..., **kwargs: Any
) -> Any: ...
@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args: Any, **kwargs: Any) -> None: ...
def autograd_function_backward_rewritten(original_backward: Any) -> Any: ...
