import re
import sys
import traceback
import warnings
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import yaml
from sentencepiece import SentencePieceProcessor

if TYPE_CHECKING:
    from wetext import Normalizer

_PUNCTUATION_MARKS_TOKENS: Final[Sequence[str]] = [
    ".",
    "!",
    "?",
    "▁.",
    "▁?",
    "▁...",  # ellipsis
]

_CJK_RANGE_PATTERN: Final = (
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)

_PINYIN_TONE_PATTERN: Final = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
"""
Matches Pinyin tone formats: pinyin + digit (tones 1-5, where 5 represents the neutral tone).
Examples: xuan4, jve2, ying1, zhong4, shang5
Non-matches: beta1, voice2
"""

_NAME_PATTERN: Final = re.compile(r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}", re.IGNORECASE)
"""
Matches person names in formats: Chinese·Chinese or Chinese·Chinese-Chinese.
Examples: 克里斯托弗·诺兰 (Christopher Nolan), 约瑟夫·高登-莱维特 (Joseph Gordon-Levitt).
"""

_TECH_TERM_PATTERN: Final = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")
"""
Matches technical terms. Format: Starts with a letter + (letters or digits)* + (-letters or digits)+
Examples: GPT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2
Must start with a letter to avoid matching pure numbers (e.g., phone numbers like 135-4567-8900).
Used to protect hyphenated structures, preventing Chinese normalizers from parsing hyphens as minus signs (e.g., "minus five").
"""

_ENGLISH_CONTRACTION_PATTERN: Final = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"
"""
Matches common English 's contractions, intended only for replacement with "is".
Does not match all instances of 's (e.g., possessives).
"""

_CHAR_REP_MAP: Final[Mapping[str, str]] = {
    "：": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": " ",
    "·": "-",
    "、": ",",
    "...": "…",
    ",,,": "…",
    "，，，": "…",
    "……": "…",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
    ":": ",",
}
_ZH_CHAR_REP_MAP: Final[Mapping[str, str]] = {"$": ".", **_CHAR_REP_MAP}


def _tokenize_by_CJK_char(line: str, do_upper_case: bool = True) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:

      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    chars = re.split(_CJK_RANGE_PATTERN, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


class TextNormalizer:
    _zh_normalizer: Normalizer | None = None
    _en_normalizer: Normalizer | None = None
    enable_glossary: bool

    def __init__(self, enable_glossary: bool = False) -> None:
        self.enable_glossary = enable_glossary
        # Terminology glossary: users can customize how domain/technical terms are read.
        # Format: {"original_term": {"en": "English pronunciation", "zh": "Chinese pronunciation"}}
        # "M.2": {"en": "M dot two", "zh": "M 二"},
        # "PCIe 5.0": {"en": "PCIE five", "zh": "PCIE 五点零"},
        # "PCIe 4.0": {"en": "PCIE four", "zh": "PCIE 四点零"},
        # "AHCI": "A H C I",
        # "TTS": "T T S",
        # "Inc.": {"en": "Ink"},
        # ".json": {"en": " dot Jay-Son", "zh": "点 Jay-Son"},
        # "C++": {"en": "C plus plus", "zh": "C 加加"},
        # "C#": "C sharp"
        # self.term_glossary = {
        #     "C++": {"en": "C plus plus", "zh": "C 加加"},
        #     "C#": "C sharp",
        #     "CMake": "C Make",
        # }
        self.term_glossary: dict[str, dict[str, str] | str] = {}
        self.load()

    @staticmethod
    def _match_email(email: str) -> bool:
        # Regex for basic email matching: alphanumerics@alphanumerics.alphas
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    def _use_chinese(self, s: str) -> bool:
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = TextNormalizer._match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        return bool(re.search(_PINYIN_TONE_PATTERN, s, re.IGNORECASE))

    def load(self) -> None:
        if self._zh_normalizer is not None and self._en_normalizer is not None:
            return
        if sys.platform != "linux":  # Mac and Windows
            from wetext import Normalizer

            self._zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self._en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh  # type: ignore
            from tn.english.normalizer import Normalizer as NormalizerEn  # type: ignore

            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = Path(__file__).resolve().parent / "tagger_cache"
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
                (cache_dir / ".gitignore").write_text("*\n")
            self._zh_normalizer = NormalizerZh(
                cache_dir=str(cache_dir), remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self._en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        if not self._zh_normalizer or not self._en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self._use_chinese(text):
            text = re.sub(_ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            # Apply glossary terms (highest priority, before all protections)
            if self.enable_glossary:
                text = self._apply_glossary_terms(text, lang="zh")
            # Protect technical terms (e.g., GPT-5-nano) to prevent incorrect processing by the Chinese normalizer
            replaced_text, tech_list = TextNormalizer._save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = TextNormalizer._save_pinyin_tones(replaced_text)

            replaced_text, original_name_list = TextNormalizer._save_names(replaced_text)
            try:
                result = self._zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # Restore names
            result = TextNormalizer._restore_names(result, original_name_list)
            # Restore pinyin tones
            result = TextNormalizer._restore_pinyin_tones(result, pinyin_list)
            # Restore technical terms
            result = TextNormalizer._restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(re.escape(p) for p in _ZH_CHAR_REP_MAP))
            result = pattern.sub(lambda x: _ZH_CHAR_REP_MAP[x.group()], result)
        else:
            try:
                text = re.sub(_ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                # Apply glossary terms (highest priority, before all protections)
                if self.enable_glossary:
                    text = self._apply_glossary_terms(text, lang="en")
                # Protect technical terms (e.g., GPT-5-Nano) to prevent incorrect processing by the English normalizer
                replaced_text, tech_list = TextNormalizer._save_tech_terms(text)
                result = self._en_normalizer.normalize(replaced_text)
                # Restore technical terms
                result = TextNormalizer._restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in _CHAR_REP_MAP))
            result = pattern.sub(lambda x: _CHAR_REP_MAP[x.group()], result)
        return result

    @staticmethod
    def _correct_pinyin(pinyin: str) -> str:
        """
        Convert the finals 'u'/'ü' in pinyin starting with j/q/x to 'v'.
        Example: ju -> jv, que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # Match pinyin starting with j/q/x where the finals are u/ü
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    @staticmethod
    def _save_names(original_text: str) -> tuple[str, list[str] | None]:
        """
        Replace names with placeholders <n_a>, <n_b>, ...
        Example: 克里斯托弗·诺兰 -> <n_a>
        """
        # Names
        original_name_list = cast(list[str], _NAME_PATTERN.findall(original_text))
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list({"".join(n) for n in original_name_list})
        transformed_text = original_text
        # Replace placeholders <n_a>, <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    @staticmethod
    def _restore_names(normalized_text: str, original_name_list: Sequence[str] | None) -> str:
        """
        Restore person names back to the original text.
        Example: <n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # Replace placeholders <n_a>, <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    @staticmethod
    def _save_tech_terms(original_text: str) -> tuple[str, list[str] | None]:
        """
        Protect hyphens in technical terms to prevent them from being parsed as minus signs by the Chinese normalizer.
        Strategy: Replace hyphens in terms with a special placeholder <H>, while numbers can still be processed normally.
        Example: GPT-5-nano -> GPT<H>5<H>nano, then 5 is converted to 五
        Finally restored to: GPT-五-nano
        """
        original_tech_list = cast(list[str], _TECH_TERM_PATTERN.findall(original_text))
        if len(original_tech_list) == 0:
            return (original_text, None)

        # Remove duplicates and sort by length in descending order (to avoid issues caused by replacing shorter matches first)
        original_tech_list = sorted(set(original_tech_list), key=len, reverse=True)
        transformed_text = original_text

        # Replace hyphens in terms with the placeholder <H>
        for term in original_tech_list:
            # Convert GPT-5-nano -> GPT<H>5<H>nano
            protected_term = term.replace("-", "<H>")
            transformed_text = transformed_text.replace(term, protected_term)

        return transformed_text, original_tech_list

    @staticmethod
    def _restore_tech_terms(normalized_text: str, original_tech_list: Sequence[str] | None) -> str:
        """
        Restore hyphens in technical terms.
        Replace placeholder <H> back to hyphen '-'.
        Also remove any extra whitespace the normalizer may have added around the placeholder.
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text

        # Remove optional whitespace around <H>, then restore to '-'.
        # Patterns handled: " <H> " -> "-", " <H>" -> "-", "<H> " -> "-", "<H>" -> "-"
        return re.sub(r"\s*<H>\s*", "-", normalized_text)

    def _apply_glossary_terms(self, text: str, lang: str = "zh") -> str:
        """
        Apply glossary terms, replacing technical terms with their pronunciation in the specified language.

        Args:
            text: Text to process.
            lang: Language type, "zh" or "en".

        Returns:
            Processed text.

        Example:
            "M.2 NVMe SSD" -> (zh) "M 二 NVMe SSD"
            "M.2 NVMe SSD" -> (en) "M dot two NVMe SSD"
        """
        if not self.term_glossary:
            return text

        # Sort terms by length in descending order to avoid shorter terms matching before longer ones.
        # For example: "PCIe 5.0" should match before "PCIe".
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)

        @lru_cache(maxsize=42)
        def get_term_pattern(term: str) -> re.Pattern[str]:
            return re.compile(re.escape(term), re.IGNORECASE)

        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            # Case-insensitive replacement via regex.
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)

        return transformed_text

    def load_glossary_from_yaml(self, glossary_path: Path) -> bool:
        """
        Load glossary terms from a YAML file.

        Args:
            glossary_path: Path to the YAML file.

        Example:
            normalizer.load_glossary_from_yaml("checkpoints/glossary.yaml")

        YAML file format:
            M.2:
              en: M dot two
              zh: M 二
            NVMe: N-V-M-E  # Same pronunciation for both Chinese and English
        """
        if glossary_path and Path(glossary_path).exists():
            with glossary_path.open(encoding="utf-8") as f:
                external_glossary = cast(dict[str, dict[str, str] | str] | object, yaml.safe_load(f))
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_glossary_to_yaml(self, glossary_path: Path) -> None:
        """
        Save the terminology glossary to a YAML file.

        Args:
            glossary_path: Path to the YAML file.
        """
        with glossary_path.open("w", encoding="utf-8") as f:
            yaml.dump(self.term_glossary, f, allow_unicode=True, default_flow_style=False)

    @staticmethod
    def _save_pinyin_tones(original_text: str) -> tuple[str, Sequence[str] | None]:
        """
        Replace pinyin tone forms with placeholders: <pinyin_a>, <pinyin_b>, ...
        Example: xuan4 -> <pinyin_a>
        """
        # Initial+final + tone digit.
        origin_pinyin_pattern = re.compile(_PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = cast(list[str], re.findall(origin_pinyin_pattern, original_text))
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list({"".join(p) for p in original_pinyin_list})
        transformed_text = original_text
        # Replace with placeholders <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        return transformed_text, original_pinyin_list

    @staticmethod
    def _restore_pinyin_tones(normalized_text: str, original_pinyin_list: Sequence[str] | None) -> str:
        """
        Restore pinyin tone digits (1-5) back to the original pinyin.
        Example: <pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # Replace placeholders <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = TextNormalizer._correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        return transformed_text


class TextTokenizer:
    _vocab_file: Path
    _normalizer: TextNormalizer
    _sp_model: SentencePieceProcessor

    def __init__(self, vocab_file: Path, normalizer: TextNormalizer) -> None:
        self._vocab_file = vocab_file
        self._normalizer = normalizer

        if not self._vocab_file.exists():
            raise ValueError(f"vocab_file {self._vocab_file} does not exist")
        if self._normalizer:
            self._normalizer.load()
        # Load vocabulary/model.
        self._sp_model = SentencePieceProcessor(model_file=str(self._vocab_file))

    @property
    def unk_token_id(self) -> int:
        return self._sp_model.unk_id()

    def convert_tokens_to_ids(self, tokens: Sequence[str] | str) -> list[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self._sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> list[str]:
        return self._encode(text)

    def _encode(self, text: str) -> list[str]:
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self._sp_model.Encode(text, out_type=str)
        # Preprocess
        if self._normalizer:
            text = self._normalizer.normalize(text)
        text = _tokenize_by_CJK_char(text)
        return self._sp_model.Encode(text, out_type=str)

    @staticmethod
    def split_segments(
        tokenized: Sequence[str], max_text_tokens_per_segment: int = 120, quick_streaming_tokens: int = 0
    ) -> list[list[str]]:
        return _split_segments_by_token(
            tokenized,
            _PUNCTUATION_MARKS_TOKENS,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )


def _split_segments_by_token(
    tokenized_str: Sequence[str],
    split_tokens: Sequence[str],
    max_text_tokens_per_segment: int,
    quick_streaming_tokens: int = 0,
) -> list[list[str]]:
    """
    Further split the tokenized result by specific tokens.
    """
    # Handle special cases
    if len(tokenized_str) == 0:
        return []
    segments: list[list[str]] = []
    current_segment: list[str] = []
    current_segment_tokens_len = 0
    for i in range(len(tokenized_str)):
        token = tokenized_str[i]
        current_segment.append(token)
        current_segment_tokens_len += 1
        if not ("," in split_tokens or "▁," in split_tokens) and ("," in current_segment or "▁," in current_segment):
            # If the current tokens contain ',', split by ','
            sub_segments = _split_segments_by_token(
                current_segment,
                [",", "▁,"],
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                quick_streaming_tokens=quick_streaming_tokens,
            )
        elif "-" not in split_tokens and "-" in current_segment:
            # If there is no ',', split by '-'
            sub_segments = _split_segments_by_token(
                current_segment,
                ["-"],
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                quick_streaming_tokens=quick_streaming_tokens,
            )
        elif current_segment_tokens_len <= max_text_tokens_per_segment:
            if token in split_tokens and current_segment_tokens_len > 2:
                if i < len(tokenized_str) - 1 and tokenized_str[i + 1] in {"'", "▁'"}:
                    # If the next token is ''', do not split
                    current_segment.append(tokenized_str[i + 1])
                    i += 1
                segments.append(current_segment)
                current_segment = []
                current_segment_tokens_len = 0
            continue
        # If the current tokens length exceeds the maximum limit
        else:
            # Split by length
            sub_segments: list[list[str]] = []
            for j in range(0, len(current_segment), max_text_tokens_per_segment):
                if j + max_text_tokens_per_segment < len(current_segment):
                    sub_segments.append(current_segment[j : j + max_text_tokens_per_segment])
                else:
                    sub_segments.append(current_segment[j:])
            warnings.warn(
                f"The tokens length of segment exceeds limit: {max_text_tokens_per_segment}, "
                + f"Tokens in segment: {current_segment}."
                + "Maybe unexpected behavior",
                RuntimeWarning,
            )
        segments.extend(sub_segments)
        current_segment = []
        current_segment_tokens_len = 0
    if current_segment_tokens_len > 0:
        assert current_segment_tokens_len <= max_text_tokens_per_segment
        segments.append(current_segment)
    # If adjacent segments together are shorter than the max limit,
    # and total tokens so far exceed quick_streaming_tokens, merge them.
    merged_segments: list[list[str]] = []
    total_token = 0
    for segment in segments:
        total_token += len(segment)
        if len(segment) == 0:
            continue
        if len(merged_segments) == 0:
            merged_segments.append(segment)
        elif (
            len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment
            and total_token > quick_streaming_tokens
        ) or len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
            merged_segments[-1] += segment
        else:
            merged_segments.append(segment)
    return merged_segments
