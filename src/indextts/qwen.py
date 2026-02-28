import json
import logging
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Final, cast

import torch
import transformers
from torch import Tensor

logger = logging.getLogger(__name__)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min(value, max_val), min_val)


PROMPT: Final = "文本情感分类"
CN_KEY_TO_EN: Final[Mapping[str, str]] = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    # NOTE: QwenEmotion's text analysis cannot reliably distinguish "低落" (melancholic)
    # from "悲伤" (sad) — the model maps both to "sad" even when the exact word "低落" is
    # present.  The ``MELANCHOLIC_WORDS`` set below implements a post-hoc workaround:
    # whenever one of those words appears in the input, the "悲伤" and "低落" scores are
    # swapped so the melancholic emotion slot is activated instead of the sad one.
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}
DESIRED_VECTOR_ORDER: Final[Sequence[str]] = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
MELANCHOLIC_WORDS: Final[Collection[str]] = {
    # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
    # to become "低落" (melancholic) instead, to fix limitations mentioned above.
    "低落",
    "melancholy",
    "melancholic",
    "depression",
    "depressed",
    "gloomy",
}
MAX_SCORE: Final = 1.2
MIN_SCORE: Final = 0.0


class QwenEmotion:
    """Qwen-based emotion classifier that converts a text input into emotion score vectors.

    Sends the user's text to a fine-tuned Qwen3 model which returns a JSON object mapping
    Chinese emotion labels to float scores.  The scores are then re-ordered, normalised,
    and clamped into the eight emotion slots expected by IndexTTS2.

    The eight output emotions (in order) are:
    ``happy``, ``angry``, ``sad``, ``afraid``, ``disgusted``,
    ``melancholic``, ``surprised``, ``calm``.

    Args:
        model_path: HuggingFace model ID or local path to the Qwen emotion model.
    """

    model: transformers.Qwen3ForCausalLM
    tokenizer: transformers.Qwen2Tokenizer

    def __init__(self, model_path: str) -> None:
        self.tokenizer = transformers.Qwen2Tokenizer.from_pretrained(model_path)
        self.model = transformers.Qwen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype="float16",  # "auto"
            device_map="auto",
        )

    def convert(self, content: dict[str, float]) -> dict[str, float]:
        """Convert a raw Chinese-keyed score dict into the ordered English emotion vector dict.

        Args:
            content: Mapping of Chinese emotion labels to raw float scores,
                as produced by parsing the model's JSON output.

        Returns:
            Ordered dict with English emotion keys in the canonical IndexTTS2 order,
            values clamped to ``[MIN_SCORE, MAX_SCORE]``.
        """
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {
            CN_KEY_TO_EN[cn_key]: _clamp(content.get(cn_key, 0.0), MIN_SCORE, MAX_SCORE)
            for cn_key in DESIRED_VECTOR_ORDER
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            logger.info("No emotions detected; using default calm/neutral voice.")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input: str) -> dict[str, float]:
        """Run the Qwen emotion classifier on *text_input* and return an emotion score dict.

        Args:
            text_input: The text whose emotional content should be classified.

        Returns:
            Ordered dict mapping English emotion names to float scores in
            ``[MIN_SCORE, MAX_SCORE]``, in the canonical IndexTTS2 order.
        """
        messages = [{"role": "system", "content": f"{PROMPT}"}, {"role": "user", "content": f"{text_input}"}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert not isinstance(text, (transformers.BatchEncoding, list))
        model_inputs = cast(Mapping[str, Tensor], self.tokenizer([text], return_tensors="pt").to(self.model.device))

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,  # pyright: ignore
            max_new_tokens=2**15,
            pad_token_id=self.model.config.eos_token_id,
        )
        assert isinstance(generated_ids, torch.Tensor)
        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(cast(list[int], output_ids[index:]), skip_special_tokens=True)
        assert isinstance(content, str)

        # decode the JSON emotion detections as a dictionary
        try:
            content = cast(dict[str, float], json.loads(content))
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            content = {m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)}

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in MELANCHOLIC_WORDS):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

        return self.convert(content)
