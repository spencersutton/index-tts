import pytest
from pathlib import Path
from indextts.infer_v2 import IndexTTS2

# Test skeletons for index-tts main functionality.
# TODO: implement tests by importing the real modules/functions from the project.


def test_main_entry_point_runs(monkeypatch, tmp_path):
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    text = "Hi!"
    tts.infer(
        spk_audio_prompt="/Users/spencer/projects/ai-scripts/catherine_laura_bailey_sample.mp3",
        text=text,
        output_path="outputs/gen.wav",
        verbose=True,
    )
