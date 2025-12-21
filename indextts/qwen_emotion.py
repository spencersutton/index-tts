import json
import re
from pathlib import Path
from typing import Final, cast

from torch import Tensor
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM
from transformers.generation.utils import GenerateOutput

MELANCHOLIC_WORDS: Final[set[str]] = {
    # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
    # to become "低落" (melancholic) instead, to fix limitations mentioned above.
    "低落",
    "melancholy",
    "melancholic",
    "depression",
    "depressed",
    "gloomy",
}
PROMPT: Final[str] = "文本情感分类"
CN_KEY_TO_EN: Final[dict[str, str]] = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    # TODO: the "低落" (melancholic) emotion will always be mapped to
    # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
    # difference between those emotions even if user writes exact words.
    # SEE: `MELANCHOLIC_WORDS` for current workaround.
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}
DESIRED_VECTOR_ORDER: Final[list[str]] = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
MAX_SCORE: Final[float] = 1.2
MIN_SCORE: Final[float] = 0.0


def clamp_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, value))


class QwenEmotion:
    model_path: Path
    _tokenizer: Qwen2Tokenizer | None = None
    _model: Qwen3ForCausalLM | None = None

    @property
    def model(self) -> Qwen3ForCausalLM:
        if self._model is not None:
            return self._model
        self._model = Qwen3ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="float16",  # "auto"
            device_map="auto",
        )
        if self._model is None:
            raise ValueError("Failed to load Qwen3ForCausalLM model.")
        return self._model

    @property
    def tokenizer(self) -> Qwen2Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        self._tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        if self._tokenizer is None:
            raise ValueError("Failed to load Qwen2Tokenizer tokenizer.")
        return self._tokenizer

    def __init__(self, model_dir: Path) -> None:
        self.model_path = model_dir

    def convert(self, content: dict[str, float]) -> dict[str, float]:
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {CN_KEY_TO_EN[cn_key]: clamp_score(content.get(cn_key, 0.0)) for cn_key in DESIRED_VECTOR_ORDER}

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input: str) -> dict[str, float]:
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{text_input}"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(text, str)
        model_inputs = self.tokenizer([text], return_tensors="pt")
        # conduct text completion
        generated_ids = cast(
            GenerateOutput | Tensor,
            self.model.generate(
                **model_inputs,
                max_new_tokens=32768,
                pad_token_id=self.tokenizer.eos_token_id,
            ),
        )
        input_ids = model_inputs.input_ids
        assert isinstance(generated_ids, Tensor)
        output_ids = generated_ids[0][len(input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = cast(dict[str, float], json.loads(content))
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            content = {
                m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', str(content))
            }

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in MELANCHOLIC_WORDS):
            content["悲伤"], content["低落"] = (
                content.get("低落", 0.0),
                content.get("悲伤", 0.0),
            )

        return self.convert(content)
