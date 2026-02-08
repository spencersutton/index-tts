from typing import override

import torch
from torch import Tensor, nn

from indextts.util import patch_call

DIM = 1280


class LearnedPositionEmbeddings(nn.Module):
    emb: nn.Embedding

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, DIM)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(std=0.02)

    @override
    def forward(self, x: int) -> Tensor:
        return self.emb(torch.arange(x, device=self.emb.weight.device))

    def get_fixed_embedding(self, index: int) -> Tensor:
        return self.emb(torch.tensor([index], device=self.emb.weight.device)).unsqueeze(0)

    @patch_call(forward)
    def __call__(self) -> None: ...
