from __future__ import annotations

from collections.abc import Sequence
from typing import override

from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.commons import sequence_mask
from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential[nn.Module]

    def __init__(
        self,
        channels: int,
        sampling_ratios: Sequence[int],
        in_channels: int,  # only applies to continuous input
        codebook_size: int = 1024,  # for discrete only
        out_channels: int | None = None,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        self.model = nn.Sequential()
        for _ in sampling_ratios:
            module = nn.Conv1d(channels, channels, 3, 1, 1)
            norm = nn.GroupNorm(groups, channels)
            act = nn.Mish()
            self.model.extend([module, norm, act])
        self.model.append(nn.Conv1d(channels, out_channels, 1, 1))

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
