from typing import override

import torch
from torch import Tensor, nn

from indextts.util import patch_call


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    @override
    def forward(self, x: Tensor) -> Tensor:
        sl = x.shape[1]
        pos = torch.arange(sl, device=x.device, dtype=torch.long)
        return self.emb(pos)

    @patch_call(forward)
    def __call__(self) -> None: ...

    def get_fixed_embedding(self, ind: int) -> Tensor:
        idx = torch.tensor([ind], device=self.emb.weight.device, dtype=torch.long)
        return self.emb(idx).unsqueeze(0)
