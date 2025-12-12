# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from packaging import version
from torch import Tensor, einsum, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

warning_printed = False


class _EfficientAttentionConfig(NamedTuple):
    enable_flash: bool
    enable_math: bool
    enable_mem_efficient: bool


# main class
class _Attend(nn.Module):
    if TYPE_CHECKING:
        mask: Tensor | None = None

    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse("2.0.0")), (
            "in order to use flash attention, you must be using pytorch 2.0 or above"
        )

        # determine efficient attention configs for cuda and cpu
        self.config = _EfficientAttentionConfig
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            if not warning_printed:
                print("A100 GPU detected, using flash attention always")
            self.cuda_config = self.config(True, False, False)
        else:
            if not warning_printed:
                print("Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda")
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n: int, device: torch.device) -> Tensor:
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        _batch, heads, q_len, _dim = q.shape
        _k_len, is_cuda = k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        assert config is not None
        backends: list[SDPBackend] = []
        if config.enable_flash:
            backends.append(SDPBackend.FLASH_ATTENTION)
        if config.enable_math:
            backends.append(SDPBackend.MATH)
        if config.enable_mem_efficient:
            backends.append(SDPBackend.EFFICIENT_ATTENTION)

        with sdpa_kernel(backends):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension.
        """
        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        return einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)


def _sequential(*mods: nn.Module | None) -> nn.Sequential:
    return nn.Sequential(*(m for m in mods if m is not None))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, scale: bool = True, dim_cond: int | None = None) -> None:
        super().__init__()
        self.cond = dim_cond is not None
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if dim_cond is not None else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        gamma = self.gamma if self.gamma is not None else 1
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert cond is not None
        assert self.to_gamma_beta is not None
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = (rearrange(t, "b d -> b 1 d") for t in (gamma, beta))
        return out * gamma + beta


class _CausalConv1d(nn.Conv1d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        causal_padded_x = F.pad(input, [self.causal_padding, 0], value=0.0)
        return super().forward(causal_padded_x)


class _GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def _feed_forward(dim: int, mult: int = 4, causal_conv: bool = False) -> nn.Sequential:
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            _CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )

    return _sequential(nn.Linear(dim, dim_inner * 2), _GEGLU(), conv, nn.Linear(dim_inner, dim))


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dim_context: int | None = None,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        dim_context = dim_context if dim_context is not None else dim

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    _Attention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        use_flash=use_flash_attn,
                        cross_attn_include_queries=True,
                    ),
                    _feed_forward(dim=dim, mult=ff_mult),
                ])
            )

        self.norm = _RMSNorm(dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for item in self.layers:
            attn, ff = cast(nn.ModuleList, item)
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_context: int | None = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        use_flash: bool = False,
        cross_attn_include_queries: bool = False,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = dim_context if dim_context is not None else dim

        self.attend = _Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        h, has_context = self.heads, context is not None

        context = context if context is not None else x

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
