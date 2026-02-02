from typing import override

from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential
    content_in_proj: nn.Linear

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(4):
            self.model.extend([nn.Conv1d(512, 512, kernel_size=3, padding=1), nn.GroupNorm(1, 512), nn.Mish()])
        self.model.append(nn.Conv1d(512, 512, kernel_size=1))

        self.content_in_proj = nn.Linear(1024, 512)

    @override
    def forward(self, x: Float[Tensor, "b t c"], ylens: int) -> Tensor:
        """Project to channels, resample in time, then refine and mask."""
        x = self.content_in_proj(x)  # (B, T, C)

        x = x.mT.contiguous()  # (B, C, T)
        x = F.interpolate(x, size=ylens)

        return self.model(x).mT.contiguous()  # (B, T, C)

    @patch_call(forward)
    def __call__(self) -> None: ...
