from pathlib import Path

import pytest
import torch

from indextts.utils.maskgct.models.codec.kmeans.vocos import ConvNeXtBlock, VocosBackbone


def test_stable_result(regen):
    artifact_path = Path(__file__).parent / "artifacts" / "vocos_backbone.pt"
    m = VocosBackbone(input_channels=2, dim=8, intermediate_dim=16, num_layers=2).eval()
    x = torch.randn(2, 2, 20)
    result = m(x)
    if regen or not artifact_path.exists():
        torch.save(result, artifact_path)
        pytest.skip("Regenerated artifact.")
    expected = torch.load(artifact_path)

    for r, e in zip(result, expected):
        assert torch.allclose(r, e)


def test_stable_result_convnextblock(regen):
    artifact_path = Path(__file__).parent / "artifacts" / "vocos_convnext.pt"
    m = ConvNeXtBlock(dim=4, intermediate_dim=8, layer_scale_init_value=0.5).eval()
    x = torch.randn(2, 4, 7)
    result = m(x)
    if regen or not artifact_path.exists():
        torch.save(result, artifact_path)
        pytest.skip("Regenerated artifact.")
    expected = torch.load(artifact_path)

    for r, e in zip(result, expected):
        assert torch.allclose(r, e)
