from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.commons import sequence_mask
from indextts.util import patch_call


class InterpolateRegulator(nn.Module):
    model: nn.Sequential

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
        model = nn.ModuleList([])
        for _ in sampling_ratios:
            module = nn.Conv1d(channels, channels, 3, 1, 1)
            norm = nn.GroupNorm(groups, channels)
            act = nn.Mish()
            model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        self.n_codebooks = 1
        self.content_in_proj = nn.Linear(in_channels, channels)

    def forward(
        self,
        x: Tensor,
        ylens: Tensor,
        n_quantizers: int = 1,
        f0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:

        x = self.content_in_proj(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(
            x.transpose(1, 2).contiguous(),
            size=int(ylens.max()),
            mode="nearest",
        )

        model_output = cast(Tensor, self.model(x))
        out = model_output.transpose(1, 2).contiguous()
        return out * mask, ylens, None, None, None

    @patch_call(
        forward,
        tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None],
    )
    def __call__(self) -> None: ...
