import torch
from collections.abc import Sequence
from torch.types import _TensorOrTensors, _TensorOrTensorsOrGradEdge, _size
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from .function import Function, NestedIOFunction
from .grad_mode import enable_grad, inference_mode, no_grad, set_grad_enabled, set_multithreading_enabled
from .gradcheck import gradcheck, gradgradcheck
from .variable import Variable
from typing import TypeAlias

__all__ = [
    "Function",
    "NestedIOFunction",
    "Variable",
    "backward",
    "detect_anomaly",
    "enable_grad",
    "grad",
    "grad_mode",
    "gradcheck",
    "gradgradcheck",
    "inference_mode",
    "no_grad",
    "set_detect_anomaly",
    "set_grad_enabled",
    "set_multithreading_enabled",
    "variable",
]
_OptionalTensor: TypeAlias = torch.Tensor | None
_ShapeorNestedShape: TypeAlias = _size | Sequence[_size] | torch.Tensor

def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: _TensorOrTensors | None = ...,
    retain_graph: bool | None = ...,
    create_graph: bool = ...,
    grad_variables: _TensorOrTensors | None = ...,
    inputs: _TensorOrTensorsOrGradEdge | None = ...,
) -> None: ...
def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: _TensorOrTensors | None = ...,
    retain_graph: bool | None = ...,
    create_graph: bool = ...,
    only_inputs: bool = ...,
    allow_unused: bool | None = ...,
    is_grads_batched: bool = ...,
    materialize_grads: bool = ...,
) -> tuple[torch.Tensor, ...]: ...
def variable(*args, **kwargs): ...

if not torch._C._autograd_init(): ...
is_multithreading_enabled = ...
is_view_replay_enabled = ...
