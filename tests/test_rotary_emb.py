"""Tests and (optional) micro-benchmarks for rotary embeddings.

These tests focus on the internal helper `_apply_rotary_emb` used by the
`gpt_fast` implementation.

Notes:
- `_apply_rotary_emb` is decorated with `torch.compile`, so parametrization is
  intentionally kept small to avoid triggering many compilation specializations.
- The benchmark is opt-in and skipped unless `INDEXTTS_RUN_BENCHMARKS=1`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# `_apply_rotary_emb` is decorated with `torch.compile`. Some environments
# (e.g., missing/unsupported toolchains) can cause Inductor compilation to fail.
# For test robustness, let Dynamo fall back to eager rather than hard-failing.
try:  # pragma: no cover
    import torch._dynamo  # type: ignore

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

# Add project root to sys.path (repo tests follow this pattern)
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.s2mel.modules.gpt_fast.model import _apply_rotary_emb, _precompute_freqs_cis


def _apply_rotary_emb_reference(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Eager reference implementation matching `_apply_rotary_emb` semantics."""
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def _allclose_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    # dtype-aware tolerances
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    return 2e-4, 2e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_apply_rotary_emb_matches_reference(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Keep shapes small to avoid excessive compile specialization.
    bsz, seqlen, n_heads, head_dim = 2, 7, 3, 16

    # Choose dtypes that are sensible per device.
    dtypes: list[torch.dtype]
    if device == "cuda":
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
    else:
        # float16 on CPU is supported in PyTorch but often slow/limited; keep it simple.
        dtypes = [torch.bfloat16, torch.float32]

    for dtype in dtypes:
        torch.manual_seed(0)
        x = torch.randn(bsz, seqlen, n_heads, head_dim, device=device, dtype=dtype)
        freqs_cis = _precompute_freqs_cis(seqlen, head_dim, base=10000).to(device=device)

        y = _apply_rotary_emb(x, freqs_cis)
        y_ref = _apply_rotary_emb_reference(x, freqs_cis)

        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert y.device == x.device

        rtol, atol = _allclose_for_dtype(dtype)
        assert torch.allclose(y, y_ref, rtol=rtol, atol=atol)


def test_apply_rotary_emb_preserves_pairwise_norm_cpu_float32() -> None:
    # Rotation should preserve L2 norm for each 2D pair (up to floating error).
    device = "cpu"
    dtype = torch.float32

    bsz, seqlen, n_heads, head_dim = 2, 5, 4, 32
    torch.manual_seed(0)
    x = torch.randn(bsz, seqlen, n_heads, head_dim, device=device, dtype=dtype)
    freqs_cis = _precompute_freqs_cis(seqlen, head_dim, base=10000).to(device=device)

    y = _apply_rotary_emb(x, freqs_cis)

    x2 = x.float().reshape(*x.shape[:-1], -1, 2)
    y2 = y.float().reshape(*y.shape[:-1], -1, 2)

    x_norm2 = (x2 * x2).sum(dim=-1)
    y_norm2 = (y2 * y2).sum(dim=-1)

    assert torch.allclose(x_norm2, y_norm2, rtol=3e-4, atol=3e-4)


def test_apply_rotary_emb_backward_matches_reference_cpu_float32() -> None:
    # `_apply_rotary_emb` includes a float-cast internally; compare gradients to the
    # reference implementation that matches that behavior.
    device = "cpu"
    dtype = torch.float32

    bsz, seqlen, n_heads, head_dim = 2, 4, 2, 16
    torch.manual_seed(0)

    x = torch.randn(bsz, seqlen, n_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    freqs_cis = _precompute_freqs_cis(seqlen, head_dim, base=10000).to(device=device)

    y = _apply_rotary_emb(x, freqs_cis)
    y_ref = _apply_rotary_emb_reference(x_ref, freqs_cis)

    loss = y.sum()
    loss_ref = y_ref.sum()

    loss.backward()
    loss_ref.backward()

    assert x.grad is not None
    assert x_ref.grad is not None
    assert torch.allclose(x.grad, x_ref.grad, rtol=2e-4, atol=2e-4)


def test_apply_rotary_emb_odd_head_dim_raises() -> None:
    # head_dim must be even (pairs of 2) due to the reshape(..., -1, 2)
    bsz, seqlen, n_heads, head_dim = 1, 3, 2, 15
    x = torch.randn(bsz, seqlen, n_heads, head_dim, dtype=torch.float32)
    freqs_cis = _precompute_freqs_cis(seqlen, head_dim + 1, base=10000)  # any valid tensor

    with pytest.raises(RuntimeError):
        _apply_rotary_emb(x, freqs_cis)


def test_apply_rotary_emb_mismatched_seq_len_raises() -> None:
    bsz, seqlen, n_heads, head_dim = 1, 5, 2, 16
    x = torch.randn(bsz, seqlen, n_heads, head_dim, dtype=torch.float32)

    # freqs computed for different length -> view() should fail
    freqs_cis = _precompute_freqs_cis(seqlen - 1, head_dim, base=10000)

    with pytest.raises(RuntimeError):
        _apply_rotary_emb(x, freqs_cis)


def test_apply_rotary_emb_benchmark(request: pytest.FixtureRequest) -> None:
    """Opt-in micro-benchmark.

    - Skipped unless `INDEXTTS_RUN_BENCHMARKS=1`.
    - Uses pytest-benchmark if available; otherwise prints a simple timing.

    This is intended to be run manually, not in CI.
    """

    if os.getenv("INDEXTTS_RUN_BENCHMARKS") != "1":
        pytest.skip("Set INDEXTTS_RUN_BENCHMARKS=1 to enable")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Representative-ish Attention Q/K/V rotary application sizes.
    bsz, seqlen, n_heads, head_dim = 1, 256, 32, 64

    torch.manual_seed(0)
    x = torch.randn(bsz, seqlen, n_heads, head_dim, device=device, dtype=dtype)
    freqs_cis = _precompute_freqs_cis(seqlen, head_dim, base=10000).to(device=device)

    # Warmup a little (also triggers compilation for the chosen specialization).
    for _ in range(3):
        y = _apply_rotary_emb(x, freqs_cis)
        if device == "cuda":
            torch.cuda.synchronize()
        del y

    def run_once() -> None:
        y = _apply_rotary_emb(x, freqs_cis)
        if device == "cuda":
            torch.cuda.synchronize()
        # Use the result so the call is not trivially removable.
        _ = y[0, 0, 0, 0].item()

    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        benchmark = None

    if benchmark is not None:
        benchmark(run_once)
        return

    # Fallback: simple wall-clock timing (still useful when pytest-benchmark isn't installed).
    n = 25
    t0 = time.perf_counter()
    for _ in range(n):
        run_once()
    t1 = time.perf_counter()

    per_iter_ms = (t1 - t0) * 1000.0 / float(n)
    print(f"_apply_rotary_emb benchmark (fallback): device={device} dtype={dtype} {per_iter_ms:.3f} ms/iter")
