from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from indextts.s2mel.modules.commons import sequence_mask

CHANNELS: Final = 512


class InterpolateRegulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = [
            layer
            for _ in range(4)
            for layer in (nn.Conv1d(CHANNELS, CHANNELS, 3, padding=1), nn.GroupNorm(1, CHANNELS), nn.Mish())
        ] + [nn.Conv1d(CHANNELS, CHANNELS, 1)]
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(2048, CHANNELS)

        self.mask_token = nn.Parameter(torch.zeros(1, CHANNELS))

        self.content_in_proj = nn.Linear(1024, CHANNELS)

    def forward(self, x: torch.Tensor, ylens: torch.Tensor):
        x: torch.Tensor = self.content_in_proj(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode="nearest")

        model_output: torch.Tensor = self.model(x)
        out = model_output.transpose(1, 2).contiguous()
        return out * mask, ylens, None, None, None
