from typing import Final, override

from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.constants import S2MEL_MODEL_DIM
from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential
    content_in_proj: nn.Linear
    dim: Final = S2MEL_MODEL_DIM

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(4):
            self.model.extend([
                nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.dim),
                nn.Mish(),
            ])
        self.model.append(nn.Conv1d(self.dim, self.dim, kernel_size=1))

        self.content_in_proj = nn.Linear(self.dim * 2, self.dim)

    @override
    def forward(self, x: Float[Tensor, "b t c"], ylens: int) -> Tensor:
        """Project to channels, resample in time, then refine and mask."""
        x = self.content_in_proj(x)  # (B, T, C)

        x = x.mT.contiguous()  # (B, C, T)
        x = F.interpolate(x, size=ylens)

        return self.model(x).mT.contiguous()  # (B, T, C)

    @patch_call(forward)
    def __call__(self) -> None: ...
