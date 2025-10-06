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
    text = "Translate for me, what is a surprise!"
    tts.infer(
        spk_audio_prompt="examples/voice_01.wav",
        text=text,
        output_path="gen.wav",
        verbose=True,
    )

    text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
    tts.infer(
        spk_audio_prompt="examples/voice_07.wav",
        text=text,
        output_path="gen.wav",
        emo_audio_prompt="examples/emo_sad.wav",
        verbose=True,
    )

    text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
    tts.infer(
        spk_audio_prompt="examples/voice_07.wav",
        text=text,
        output_path="gen.wav",
        emo_audio_prompt="examples/emo_sad.wav",
        emo_alpha=0.9,
        verbose=True,
    )
