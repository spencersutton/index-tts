from typing import override

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn

from indextts.util import patch_call


class LearnedPositionEmbeddings(nn.Module):
    emb: nn.Embedding

    def __init__(self, seq_len: int, dim: int = 1280) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(std=0.02)

    @override
    @beartype
    def forward(self, x: int) -> Float[Tensor, "seq dim"]:
        return self.emb(torch.arange(x))

    @beartype
    def get_fixed_embedding(self, index: int) -> Float[Tensor, "1 1 dim"]:
        return self.emb(torch.tensor([index]).unsqueeze(0))

    @patch_call(forward)
    def __call__(self) -> None: ...
