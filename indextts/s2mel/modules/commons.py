import torch
from torch import Tensor


@torch.compile(dynamic=True)
def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor) -> Tensor:
    t_act_part, s_act_part = torch.chunk(input_a + input_b, 2, dim=1)
    return torch.tanh(t_act_part) * torch.sigmoid(s_act_part)
