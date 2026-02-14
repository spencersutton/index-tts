from typing import Final, override

from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call

DIM: Final = 512


class InterpolateRegulator(nn.Module):
    model: nn.Sequential
    content_in_proj: nn.Linear

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(4):
            self.model.extend([nn.Conv1d(DIM, DIM, kernel_size=3, padding=1), nn.GroupNorm(1, DIM), nn.Mish()])
        self.model.append(nn.Conv1d(DIM, DIM, kernel_size=1))

        self.content_in_proj = nn.Linear(DIM * 2, DIM)

    @override
    def forward(self, x: Tensor, ylens: int) -> Tensor:
        """Project to channels, resample in time, then refine and mask."""
        x = self.content_in_proj(x)  # (B, T, C)

        x = x.mT.contiguous()  # (B, C, T)
        x = F.interpolate(x, size=ylens)

        return self.model(x).mT.contiguous()  # (B, T, C)

    @patch_call(forward)
    def __call__(self) -> None: ...
