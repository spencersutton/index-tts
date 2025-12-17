import torch
from torch import nn

from ..utils import is_accelerate_available, is_fbgemm_gpu_available, is_torch_available

if is_torch_available(): ...
if is_accelerate_available(): ...
if is_fbgemm_gpu_available(): ...
logger = ...

class FbgemmFp8Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, weight_dtype=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class FbgemmFp8Llama4TextExperts(nn.Module):
    def __init__(self, config, dtype=...) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:

        ...

def replace_with_fbgemm_fp8_linear(
    model,
    modules_to_not_convert=...,
    current_key_name=...,
    quantization_config=...,
    pre_quantized=...,
    config=...,
    tp_plan=...,
): ...
