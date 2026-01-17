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
        self.emb.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind: int, dev: torch.device) -> Tensor:
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)

    @patch_call(forward)
    def __call__(self) -> None: ...
