import pytest
import torch

from indextts.gpt.conformer.subsampling import Conv2dSubsampling2


def test_stable_result(regen, artifact):
    m = Conv2dSubsampling2(idim=6, odim=4, dropout_rate=0.0).eval()
    x = torch.randn(2, 5, 6)
    x_mask = torch.ones(2, 1, 5, dtype=torch.bool)
    result = m(x, x_mask)
    if regen or not artifact.exists():
        torch.save(result, artifact)
        pytest.skip("Regenerated artifact.")
    expected = torch.load(artifact)

    for r, e in zip(result, expected):
        assert torch.allclose(r, e)
