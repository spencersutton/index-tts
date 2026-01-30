from typing import override

from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.commons import sequence_mask
from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential
    content_in_proj: nn.Linear

    def __init__(self, channels: int = 512, groups: int = 1, num_samples: int = 4, in_channels: int = 1024) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(num_samples):
            self.model.extend([
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, channels),
                nn.Mish(),
            ])
        self.model.append(nn.Conv1d(channels, channels, kernel_size=1))

        self.content_in_proj = nn.Linear(in_channels, channels)

    @override
    def forward(self, x: Float[Tensor, "b t c"], ylens: Int[Tensor, "b"]) -> Tensor:
        """Project to channels, resample in time, then refine and mask."""
        x = self.content_in_proj(x)  # (B, T, C)
        mask = sequence_mask(ylens).unsqueeze(-1)  # (B, T, 1)

        x = x.mT.contiguous()  # (B, C, T)
        x = F.interpolate(x, size=int(ylens.max()))

        out = self.model(x).mT.contiguous()  # (B, T, C)
        return out * mask

    @patch_call(forward)
    def __call__(self) -> None: ...
