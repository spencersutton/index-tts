import os
from pathlib import Path
import pytest
import torch

from indextts.infer_v2 import IndexTTS2


def test_cli_equivalent_infer() -> None:
    voice_path = Path("outputs/mizora.ogg")
    output_path = Path("outputs/gen.wav")
    text = "This is a test."

    # torch = pytest.importorskip("torch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    use_fp16 = False
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():  # pyright: ignore[reportAttributeAccessIssue]
        device = "xpu"
    elif hasattr(torch, "mps") and torch.mps.is_available():  # pyright: ignore[reportAttributeAccessIssue]
        device = "mps"
    else:
        device = "cpu"
        use_fp16 = False

    tts = IndexTTS2(model_dir=Path("checkpoints"), use_fp16=use_fp16, device=device)
    result = tts.infer(output_path=output_path, spk_audio_prompt=voice_path, text=text)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    if isinstance(result, Path):
        assert result == output_path
