import pytest
import torch

from indextts.gpt.conformer.modules import ConvolutionModule


def test_forward_outputs_shape_and_cache():
    m = ConvolutionModule(channels=4, kernel_size=3)
    x = torch.randn(2, 7, 4)
    out, cache = m(x)
    assert out.shape == x.shape
    assert isinstance(cache, torch.Tensor)
    assert cache.shape == (0, 0, 0)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_forward_applies_mask_correctly():
    m = ConvolutionModule(channels=3, kernel_size=3)
    batch, time, channels = 2, 6, 3
    x = torch.ones(batch, time, channels)
    mask = torch.zeros(batch, 1, time, dtype=torch.bool)
    mask[:, :, :3] = True  # first half valid, second half padded
    out, _ = m(x, mask)
    # Masked (padded) time steps must be zero after the module (masked twice inside)
    assert torch.allclose(out[:, 3:, :], torch.zeros_like(out[:, 3:, :]), atol=1e-6)
    # Ensure at least some unmasked values are non-zero to avoid false positives
    assert out[:, :3, :].abs().sum() > 1e-6


def test_stable_output_on_repeated_calls():
    m = ConvolutionModule(channels=3, kernel_size=3)
    x = torch.randn(2, 5, 3)
    out1, _ = m(x)
    out2, _ = m(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_stable_result(regen, artifact):
    m = ConvolutionModule(channels=3, kernel_size=3).eval()
    x = torch.randn(2, 5, 3)
    result, _ = m(x)
    if regen or not artifact.exists():
        torch.save(result, artifact)
        pytest.skip("Regenerated artifact.")
    expected = torch.load(artifact)
    assert torch.allclose(result, expected)
