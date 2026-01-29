from indextts.infer_v2 import IndexTTS2


def test_infer_v2_loads_all_cached_modules() -> None:
    tts = IndexTTS2()

    _ = tts.gpt
    _ = tts.qwen_emo
    _ = tts.normalizer
    _ = tts.tokenizer
    _ = tts.campplus_model
    _ = tts.bigvgan
    _ = tts.semantic_codec
    _ = tts.semantic_model
    _ = tts.semantic_mean
    _ = tts.semantic_std
    _ = tts.s2mel
    _ = tts.extract_features
