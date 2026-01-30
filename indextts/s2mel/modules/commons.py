import torch
from jaxtyping import Int
from torch import Tensor, nn


def sequence_mask(length: Int[Tensor, "b"] | int, max_length: int | None = None) -> Tensor:
    length = torch.as_tensor(length)
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator

    cfm: CFM
    length_regulator: InterpolateRegulator
    gpt_layer: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

        self.cfm = CFM()
        self.length_regulator = InterpolateRegulator()
        self.gpt_layer = nn.Sequential(nn.Linear(1280, 256), nn.Linear(256, 128), nn.Linear(128, 1024))

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization.

        This method applies torch.compile to the model for significant
        performance improvements during inference.
        """
        self.cfm.enable_torch_compile()
