from pathlib import Path

from indextts.infer_v2 import IndexTTS2


def test_inference(benchmark):
    tts = IndexTTS2(
        use_cuda_kernel=False,
        use_accel=False,
        use_torch_compile=False,
    )

    benchmark(
        tts.infer,
        spk_audio_prompt=Path("/home/spencer/index-tts/outputs/mizora.ogg"),
        text="This is a test for benchmarking. It requires a decent amount of text to generate to produce good results.",
        output_path=Path("benchmark_output_with_prompt.wav"),
    )
