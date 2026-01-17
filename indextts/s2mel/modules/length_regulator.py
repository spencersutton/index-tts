from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.commons import sequence_mask
from indextts.util import patch_call

CHANNELS: Final = 512


class InterpolateRegulator(nn.Module):
    model: nn.Sequential
    content_in_proj: nn.Linear
    embedding: nn.Embedding
    mask_token: nn.Parameter

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for _ in range(4):
            self.model.extend([nn.Conv1d(CHANNELS, CHANNELS, 3, padding=1), nn.GroupNorm(1, CHANNELS), nn.Mish()])
        self.model.append(nn.Conv1d(CHANNELS, CHANNELS, 1))

        self.embedding = nn.Embedding(2048, CHANNELS)
        self.mask_token = nn.Parameter(torch.zeros(1, CHANNELS))
        self.content_in_proj = nn.Linear(1024, CHANNELS)

    def forward(self, x: Tensor, ylens: Tensor) -> Tensor:
        x = self.content_in_proj(x)  # (B, T, C)
        mask = sequence_mask(ylens).unsqueeze(-1)  # (B, T, 1)

        x = x.mT.contiguous()  # (B, C, T)
        x = F.interpolate(x, size=ylens.max(), mode="nearest")

        out = self.model(x).mT.contiguous()  # (B, T, C)
        return out * mask

    @patch_call(forward)
    def __call__(self) -> None: ...
