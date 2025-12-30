"""Optional end-to-end streaming integration test.

This exercises the real `IndexTTS2.infer(..., stream_return=True)` path.

It is skipped by default because it requires local checkpoints and can be slow.
Enable with:
  INDEXTTS_RUN_INTEGRATION=1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_streaming_inference_real(tmp_path: Path) -> None:
    if os.getenv("INDEXTTS_RUN_INTEGRATION") != "1":
        pytest.skip("Set INDEXTTS_RUN_INTEGRATION=1 to enable")

    root_dir = Path(__file__).parent.parent
    checkpoint_dir = root_dir / "checkpoints"
    config_path = checkpoint_dir / "config.yaml"

    if not checkpoint_dir.exists() or not config_path.exists():
        pytest.skip("Checkpoints/config.yaml not found")

    try:
        import torchaudio
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torchaudio not available: {e}")

    # Create a dummy wav prompt in a temp directory.
    prompt_wav = tmp_path / "temp_test_prompt.wav"
    sample_rate = 16000
    dummy_audio = torch.randn(1, sample_rate)
    torchaudio.save(str(prompt_wav), dummy_audio, sample_rate)

    try:
        from indextts.infer_v2 import IndexTTS2
    except Exception as e:  # pragma: no cover
        pytest.skip(f"IndexTTS2 import failed in this environment: {e}")

    # Prefer CPU for determinism/portability; integration users can override by editing locally.
    device = "cpu"

    tts = IndexTTS2(
        cfg_path=config_path,
        model_dir=checkpoint_dir,
        device=device,
        use_fp16=False,
        use_cuda_kernel=False,
        use_accel=False,
    )

    text = "Hello world. This is a test of streaming inference."

    generator = tts.infer(
        spk_audio_prompt=prompt_wav,
        text=text,
        output_path=None,
        stream_return=True,
        max_text_tokens_per_segment=10,
        verbose=False,
    )

    assert hasattr(generator, "__iter__")

    chunks: list[torch.Tensor] = []
    for chunk in generator:
        assert isinstance(chunk, torch.Tensor)
        assert chunk.numel() > 0
        chunks.append(chunk)

    assert len(chunks) > 0
    full_audio = torch.cat(chunks, dim=-1)
    assert full_audio.shape[-1] > 0
