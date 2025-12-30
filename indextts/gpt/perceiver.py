# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532
from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn

from indextts.util import patch_call

warning_printed = False


class _Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)


def _split_heads(x: Tensor, heads: int) -> Tensor:
    """(b, n, h*d) -> (b, h, n, d)"""
    b, n, inner = x.shape
    assert inner % heads == 0
    d = inner // heads
    return x.reshape(b, n, heads, d).permute(0, 2, 1, 3)


def _merge_heads(x: Tensor) -> Tensor:
    """(b, h, n, d) -> (b, n, h*d)"""
    b, h, n, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, n, h * d)


# main class
class Attend(nn.Module):
    if TYPE_CHECKING:
        mask: Tensor | None = None
    attn_dropout: nn.Dropout

    def __init__(self) -> None:
        super().__init__()
        self.attn_dropout = nn.Dropout(0.0)

        self.register_buffer("mask", None, persistent=False)

    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension.
        """
        scale = cast(float, q.shape[-1] ** -0.5)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        if mask is not None:
            mask = mask[:, None, None, :]
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        return einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

    @patch_call(forward)
    def __call__(self) -> None: ...


class RMSNorm(nn.Module):
    scale: float
    gamma: nn.Parameter

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.scale = float(dim**0.5)
        self.gamma = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma

    @patch_call(forward)
    def __call__(self) -> None: ...


class GEGLU(nn.Module):
    @override
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

    @patch_call(forward)
    def __call__(self) -> None: ...


class PerceiverResampler(nn.Module):
    proj_context: nn.Linear
    latents: nn.Parameter
    layers: nn.ModuleList[nn.ModuleList[nn.Module]]
    norm: RMSNorm

    def __init__(self, dim: int = 512, heads: int = 8, num_latents: int = 32) -> None:
        super().__init__()

        self.proj_context = nn.Linear(512, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        dim_inner = int(dim * 4 / 3)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim=dim, heads=heads),
                nn.Sequential(
                    nn.Linear(dim, dim_inner * 2),
                    GEGLU(),
                    nn.Linear(dim_inner, dim),
                ),
            ])
            for _ in range(2)
        ])

        self.norm = RMSNorm(dim)

    @override
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)

        for item in self.layers:
            attn, ff = item
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

    @patch_call(forward)
    def __call__(self) -> None: ...


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.scale = float(64**-0.5)
        self.heads = heads
        self.cross_attn_include_queries = True

        dim_inner = 64 * heads
        self.attend = Attend()
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    @override
    def forward(self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        h, has_context = self.heads, context is not None

        context = context if context is not None else x

        if has_context:
            context = torch.cat((x, context), dim=-2)

        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q = _split_heads(q, h)
        k = _split_heads(k, h)
        v = _split_heads(v, h)

        out = self.attend(q, k, v, mask=mask)

        out = _merge_heads(out)
        return self.to_out(out)

    @patch_call(forward)
    def __call__(self) -> None: ...
