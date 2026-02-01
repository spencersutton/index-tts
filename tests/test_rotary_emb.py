import torch

from indextts.s2mel.modules.gpt_fast.model import _apply_rotary_emb  # pyright: ignore[reportPrivateUsage]


def _apply_rotary_emb_reference(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Reference implementation matching the original code path."""

    xshaped = x.reshape(*x.shape[:-1], -1, 2)
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


@torch.no_grad()
def test_apply_rotary_emb_matches_reference() -> None:
    device = torch.device("cpu")

    # Keep sizes tiny so this test is fast and CPU-friendly.
    b, t, h, d = 2, 5, 3, 16
    d_half = d // 2

    # Build a (t, d/2, 2) freqs tensor: [cos(theta), sin(theta)]
    angles = torch.randn(t, d_half, device=device, dtype=torch.float32)

    for dtype, atol, rtol in [
        (torch.float32, 1e-6, 1e-5),
        # bfloat16 is commonly used in this project and is supported on CPU.
        (torch.bfloat16, 5e-2, 5e-2),
    ]:
        freqs_cis = torch.stack([angles.cos(), angles.sin()], dim=-1).to(dtype=dtype)

        # Make x non-contiguous on purpose.
        x_base = torch.randn(b, h, t, d, device=device, dtype=dtype)
        x = x_base.transpose(1, 2)  # (b, t, h, d), typically non-contiguous

        out_ref = _apply_rotary_emb_reference(x, freqs_cis)
        out_opt = _apply_rotary_emb(x, freqs_cis)

        assert out_ref.shape == out_opt.shape
        assert out_ref.dtype == out_opt.dtype
        assert torch.allclose(out_opt, out_ref, atol=atol, rtol=rtol)
