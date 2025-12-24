from __future__ import annotations

from collections.abc import Sequence
from typing import override

from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.commons import sequence_mask
from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential[nn.Module]
    content_in_proj: nn.Linear

    def __init__(
        self,
        channels: int,
        sampling_ratios: Sequence[int],
        in_channels: int,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in sampling_ratios:
            self.model.extend([
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.GroupNorm(groups, channels),
                nn.Mish(),
            ])
        self.model.append(nn.Conv1d(channels, channels, 1, 1))

        self.content_in_proj = nn.Linear(in_channels, channels)

    @override
    def forward(
        self,
        x: Tensor,
        ylens: Tensor,
        n_quantizers: int = 3,
    ) -> Tensor:
        x = self.content_in_proj(x)  # x in (B, T, D)

        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(
            x.transpose(1, 2).contiguous(),
            size=int(ylens.max()),
            mode="nearest",
        )

        return self.model(x).transpose(1, 2).contiguous() * mask

    @patch_call(forward)
    def __call__(self) -> None: ...
