import torch
from torch import Tensor


@torch.compile
def fused_add_tanh_sigmoid_multiply(
    input_a: Tensor,
    input_b: Tensor,
) -> Tensor:
    in_act = input_a + input_b
    t_act_part, s_act_part = torch.chunk(in_act, 2, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    return t_act * s_act


def sequence_mask(length: Tensor, max_length: Tensor | int | None = None) -> Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
