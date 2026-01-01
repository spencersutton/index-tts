import pytest
import torch

from indextts.s2mel.modules.length_regulator import InterpolateRegulator


def test_stable_result(regen, artifact):
    reg = InterpolateRegulator().eval()
    result = reg(torch.randn(2, 5, 1024), 5)
    if regen or not artifact.exists():
        torch.save(result, artifact)
        pytest.skip("Regenerated artifact.")
    expected = torch.load(artifact)
    assert torch.allclose(result, expected)
