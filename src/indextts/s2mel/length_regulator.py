from typing import override

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential
    content_in_proj: nn.Linear

    def __init__(self, dim: int = 512) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(4):
            self.model.extend([nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.GroupNorm(1, dim), nn.Mish()])
        self.model.append(nn.Conv1d(dim, dim, kernel_size=1))

        self.content_in_proj = nn.Linear(dim * 2, dim)

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch time_in in_dim"], ylens: int) -> Float[Tensor, "batch time_out dim"]:
        """Project to channels, resample in time, then refine and mask."""
        x = self.content_in_proj(x)  # (B, T, C)

        x = x.mT.contiguous()  # (B, C, T)
        x = F.interpolate(x, size=ylens)

        return self.model(x).mT.contiguous()  # (B, T, C)

    @patch_call(forward)
    def __call__(self) -> None: ...


if __name__ == "__main__":
    torch.onnx.export(
        InterpolateRegulator().eval(),
        (torch.randn(1, 10, 1024), 20),
        "length_regulator.onnx",
        input_names=["x", "ylens"],
        output_names=["output"],
        dynamic_shapes={"x": {1: torch.export.Dim("input_length")}, "ylens": None},
    )
