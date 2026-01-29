from pathlib import Path

from indextts.infer_v2 import IndexTTS2

voice_path = Path("outputs/mizora.ogg")
output_path = Path("outputs/gen.wav")
text = "This is a test."


def test_cli_equivalent_infer() -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    tts = IndexTTS2()
    result = tts.infer(output_path=output_path, spk_audio_prompt=voice_path, text=text)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    if isinstance(result, Path):
        assert result == output_path


def test_accel() -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    tts = IndexTTS2(use_accel=True)
    result = tts.infer(output_path=output_path, spk_audio_prompt=voice_path, text=text)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    if isinstance(result, Path):
        assert result == output_path
