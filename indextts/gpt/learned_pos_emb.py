from typing import override

import torch
from torch import Tensor, nn


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    @override
    def forward(self, x: Tensor) -> Tensor:
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl))

    def get_fixed_embedding(self, ind: int) -> Tensor:
        return self.emb(torch.tensor([ind])).unsqueeze(0)
