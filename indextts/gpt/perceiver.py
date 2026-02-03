# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532
from typing import cast, override

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from indextts.util import patch_call


class _Attention(nn.Module):
    to_kv: nn.Linear
    heads: int
    to_out: nn.Linear
    to_q: nn.Linear
    attend: _Attend

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads

        dim_inner = 64 * heads

        self.attend = _Attend()
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    @override
    def forward(self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        h = self.heads

        context = context if context is not None else x
        context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _Attend(nn.Module):
    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        scale = cast(float, q.shape[-1] ** -0.5)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity
        sim = torch.einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)

        # aggregate values
        return torch.einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _RMSNorm(nn.Module):
    scale: float
    gamma: nn.Parameter

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.scale = cast(float, dim**0.5)
        self.gamma = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma

    @patch_call(forward)
    def __call__(self) -> None: ...


class _GEGLU(nn.Module):
    @override
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

    @patch_call(forward)
    def __call__(self) -> None: ...


class PerceiverResampler(nn.Module):
    proj_context: nn.Linear
    latents: nn.Parameter
    layers: nn.ModuleList[nn.ModuleList[_Attention | nn.Sequential]]
    norm: _RMSNorm

    def __init__(self, dim: int, num_latents: int = 32, heads: int = 8, depth: int = 2, dim_context: int = 512) -> None:
        super().__init__()

        self.proj_context = nn.Linear(dim_context, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList()
        dim_inner = int(dim * 4 / 3)
        for _ in range(depth):
            layer = nn.ModuleList((
                _Attention(dim=dim, heads=heads),
                nn.Sequential(nn.Linear(dim, dim_inner * 2), _GEGLU(), nn.Linear(dim_inner, dim)),
            ))
            self.layers.append(layer)

        self.norm = _RMSNorm(dim)

    @override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.proj_context(x)

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        for attn, ff in self.layers:
            assert isinstance(attn, _Attention) and isinstance(ff, nn.Sequential)
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

    @patch_call(forward)
    def __call__(self) -> None: ...
