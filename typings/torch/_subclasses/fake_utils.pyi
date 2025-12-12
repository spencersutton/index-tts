import torch
from typing import Any, Callable, Union
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import TorchDispatchMode

aten = ...

def outputs_alias_inputs(outputs, inputs):  # -> bool:
    ...
def outputs_are_inputs(outputs, inputs):  # -> bool:
    ...
def output_alias_each_other(outputs):  # -> bool:
    ...
def is_sdpa_error(func, idx, e):  # -> bool:
    ...
def try_convert_fake_to_real(ten_list: list[Union[FakeTensor, Any]]) -> list[Union[FakeTensor, torch.Tensor, Any]]: ...

class CrossRefFakeMode(TorchDispatchMode):
    def __init__(
        self,
        ignore_op_fn: Union[Callable[[OpOverload], bool], None] = ...,
        *,
        check_strides=...,
        check_aliasing=...,
        only_check_ops_with_meta=...,
    ) -> None: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=...): ...
