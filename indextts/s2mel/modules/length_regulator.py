from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential[nn.Module]
    content_in_proj: nn.Linear

    def __init__(
        self,
        channels: int,
        sampling_ratios: int,
        in_channels: int,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(sampling_ratios):
            self.model.extend([
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.GroupNorm(groups, channels),
                nn.Mish(),
            ])
        self.model.append(nn.Conv1d(channels, channels, 1, 1))

        self.content_in_proj = nn.Linear(in_channels, channels)

    @override
    def forward(self, x: Tensor, ylens: int) -> Tensor:
        x = self.content_in_proj(x)  # x in (B, T, D)
        x = F.interpolate(
            x.transpose(1, 2).contiguous(),
            size=ylens,
            mode="nearest",
        )

        mask = torch.full((1, ylens, 1), True, device=x.device)
        return self.model(x).transpose(1, 2).contiguous() * mask

    @patch_call(forward)
    def __call__(self) -> None: ...
