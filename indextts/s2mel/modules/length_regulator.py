from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call

CHANNELS = 512
LAYERS = 4
IN_CHANNELS = 1024


class InterpolateRegulator(nn.Module):
    content_in_proj: nn.Linear
    model: nn.Sequential[nn.Module]

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(LAYERS):
            self.model.extend([
                nn.Conv1d(CHANNELS, CHANNELS, 3, 1, 1),
                nn.GroupNorm(1, CHANNELS),
                nn.Mish(),
            ])
        self.model.append(nn.Conv1d(CHANNELS, CHANNELS, 1, 1))
        self.content_in_proj = nn.Linear(IN_CHANNELS, CHANNELS)

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
