"""Optional inference benchmark.

This is an integration/benchmark test that requires:
- local checkpoints in `checkpoints/`
- `torchaudio` and model deps

It is skipped by default. Enable with:
  INDEXTTS_RUN_BENCHMARKS=1
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_inference_benchmark(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    if os.getenv("INDEXTTS_RUN_BENCHMARKS") != "1":
        pytest.skip("Set INDEXTTS_RUN_BENCHMARKS=1 to enable")

    root_dir = Path(__file__).parent.parent
    checkpoint_dir = root_dir / "checkpoints"
    config_path = checkpoint_dir / "config.yaml"
    prompt_wav = root_dir / "tests" / "sample_prompt.wav"

    if not checkpoint_dir.exists() or not config_path.exists():
        pytest.skip("Checkpoints/config.yaml not found")
    if not prompt_wav.exists():
        pytest.skip("tests/sample_prompt.wav not found")

    try:
        from indextts.infer_v2 import IndexTTS2
    except Exception as e:  # pragma: no cover
        pytest.skip(f"IndexTTS2 import failed in this environment: {e}")

    tts = IndexTTS2(
        cfg_path=config_path,
        model_dir=checkpoint_dir,
        device="cpu",
        use_fp16=False,
        use_cuda_kernel=False,
        use_accel=False,
    )

    out_path = tmp_path / "benchmark_output.wav"

    def run_once() -> None:
        # Avoid streaming in benchmark; write to a temp file.
        list(
            tts.infer(
                spk_audio_prompt=prompt_wav,
                text="This is a short benchmark run.",
                output_path=out_path,
                stream_return=False,
                verbose=False,
            )
        )

    # Use pytest-benchmark if installed; otherwise just run once.
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        benchmark = None

    if benchmark is not None:
        benchmark(run_once)
    else:
        run_once()
