import re
import sys
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
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


def _de_tokenized_by_CJK_char(line: str, do_lower_case: bool = False) -> str:
    """
    Example:
      input = "你 好 世 界 是 HELLO WORLD 的 中 文"
      output = "你好世界是 hello world 的中文"

    do_lower_case:
      input = "SEE YOU!"
      output = "see you!"
    """
    # replace english words in the line with placeholders
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s'-][A-Z-]+)*)", re.IGNORECASE)
    english_sents: list[str] = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # restore english sentences
    sent_placeholder_pattern = re.compile(r"(<sent_(\d+)>)")
    for i in range(len(words)):
        all_matches: list[tuple[str, str]] = sent_placeholder_pattern.findall(words[i])
        if len(all_matches) > 1:
            # restore the english word
            for h, j in all_matches:
                placeholder_index = int(j)
                words[i] = words[i].replace(h, english_sents[placeholder_index])
                if do_lower_case:
                    words[i] = words[i].lower()
    return "".join(words)


CJK_RANGE_PATTERN: Final = (
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)


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
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


PINYIN_TONE_PATTERN: Final = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
"""
Matches Pinyin tone formats: pinyin + digit (tones 1-5, where 5 represents the neutral tone).
Examples: xuan4, jve2, ying1, zhong4, shang5
Non-matches: beta1, voice2
"""

NAME_PATTERN: Final = re.compile(r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}", re.IGNORECASE)
"""
Matches person names in formats: Chinese·Chinese or Chinese·Chinese-Chinese.
Examples: 克里斯托弗·诺兰 (Christopher Nolan), 约瑟夫·高登-莱维特 (Joseph Gordon-Levitt).
"""

TECH_TERM_PATTERN: Final = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")
"""
Matches technical terms. Format: Starts with a letter + (letters or digits)* + (-letters or digits)+
Examples: GPT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2
Must start with a letter to avoid matching pure numbers (e.g., phone numbers like 135-4567-8900).
Used to protect hyphenated structures, preventing Chinese normalizers from parsing hyphens as minus signs (e.g., "minus five").
"""

ENGLISH_CONTRACTION_PATTERN: Final = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"
"""
Matches common English 's contractions, intended only for replacement with "is".
Does not match all instances of 's (e.g., possessives).
"""


class TextNormalizer:
    if TYPE_CHECKING:
        zh_normalizer: Normalizer | None
        en_normalizer: Normalizer | None
    zh_char_rep_map: Mapping[str, str]
    enable_glossary: bool
    char_rep_map: Mapping[str, str] = {
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

    def __init__(self, enable_glossary: bool = False) -> None:
        self.zh_normalizer = None
        self.en_normalizer = None
        self.zh_char_rep_map = {"$": ".", **self.char_rep_map}
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
    def match_email(email: str) -> bool:
        # Regex for basic email matching: alphanumerics@alphanumerics.alphas
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    def use_chinese(self, s: str) -> bool:
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        return bool(re.search(PINYIN_TONE_PATTERN, s, re.IGNORECASE))

    def load(self) -> None:
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if sys.platform != "linux":  # Mac and Windows
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh  # type: ignore
            from tn.english.normalizer import Normalizer as NormalizerEn  # type: ignore

            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = Path(__file__).resolve().parent / "tagger_cache"
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
                (cache_dir / ".gitignore").write_text("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=str(cache_dir), remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self.use_chinese(text):
            text = re.sub(ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            # Apply glossary terms (highest priority, before all protections)
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang="zh")
            # Protect technical terms (e.g., GPT-5-nano) to prevent incorrect processing by the Chinese normalizer
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)

            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # Restore names
            result = self.restore_names(result, original_name_list)
            # Restore pinyin tones
            result = self.restore_pinyin_tones(result, pinyin_list)
            # Restore technical terms
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                # Apply glossary terms (highest priority, before all protections)
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                # Protect technical terms (e.g., GPT-5-Nano) to prevent incorrect processing by the English normalizer
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                # Restore technical terms
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    @staticmethod
    def correct_pinyin(pinyin: str) -> str:
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
    def save_names(original_text: str) -> tuple[str, Sequence[str] | None]:
        """
        Replace names with placeholders <n_a>, <n_b>, ...
        Example: 克里斯托弗·诺兰 -> <n_a>
        """
        # Names
        original_name_list = cast(list[str], NAME_PATTERN.findall(original_text))
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
    def restore_names(normalized_text: str, original_name_list: Sequence[str] | None) -> str:
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
    def save_tech_terms(original_text: str) -> tuple[str, Sequence[str] | None]:
        """
        Protect hyphens in technical terms to prevent them from being parsed as minus signs by the Chinese normalizer.
        Strategy: Replace hyphens in terms with a special placeholder <H>, while numbers can still be processed normally.
        Example: GPT-5-nano -> GPT<H>5<H>nano, then 5 is converted to 五
        Finally restored to: GPT-五-nano
        """
        original_tech_list = cast(list[str], TECH_TERM_PATTERN.findall(original_text))
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
    def restore_tech_terms(normalized_text: str, original_tech_list: Sequence[str] | None) -> str:
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

    def apply_glossary_terms(self, text: str, lang: str = "zh") -> str:
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

    def load_glossary(self, glossary_dict: dict[str, dict[str, str] | str]) -> None:
        """
        Load external glossary terms.

        Args:
            glossary_dict: Glossary dictionary, format {"term": {"en": "English pronunciation", "zh": "Chinese pronunciation"}}

        Example:
            normalizer.load_glossary({
                "M.2": {"en": "M dot two", "zh": "M 二"},
                "PCIe": {"en": "PCIE", "zh": "PCIE"}
            })
        """
        if glossary_dict:
            self.term_glossary.update(glossary_dict)

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
                external_glossary = yaml.safe_load(f)  # pyright: ignore
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
    def save_pinyin_tones(original_text: str) -> tuple[str, Sequence[str] | None]:
        """
        Replace pinyin tone forms with placeholders: <pinyin_a>, <pinyin_b>, ...
        Example: xuan4 -> <pinyin_a>
        """
        # Initial+final + tone digit.
        origin_pinyin_pattern = re.compile(PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list({"".join(p) for p in original_pinyin_list})  # pyright: ignore
        transformed_text = original_text
        # Replace with placeholders <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text: str, original_pinyin_list: Sequence[str] | None) -> str:
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
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        return transformed_text


class TextTokenizer:
    vocab_file: Path
    normalizer: TextNormalizer
    sp_model: SentencePieceProcessor
    pre_tokenizers: list[Callable[[str], str]]

    def __init__(self, vocab_file: Path, normalizer: TextNormalizer) -> None:
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if not self.vocab_file.exists():
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # Load vocabulary/model.
        self.sp_model = SentencePieceProcessor(model_file=str(self.vocab_file))

        self.pre_tokenizers = [
            # Pre-tokenizers
            _tokenize_by_CJK_char
        ]

    @property
    def unk_token_id(self) -> int:
        return self.sp_model.unk_id()

    def convert_tokens_to_ids(self, tokens: Sequence[str] | str) -> list[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> list[str]:
        return self.encode(text)

    def encode(self, text: str) -> list[str]:
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=str)
        # Preprocess
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=str)

    def batch_encode(self, texts: list[str]) -> list[list[str]] | list[str]:
        # Preprocess
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=str)

    def decode(self, ids: list[int] | int, do_lower_case: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=str)
        return _de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_segments_by_token(
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
            if not ("," in split_tokens or "▁," in split_tokens) and (
                "," in current_segment or "▁," in current_segment
            ):
                # If the current tokens contain ',', split by ','
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment,
                    [",", "▁,"],
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    quick_streaming_tokens=quick_streaming_tokens,
                )
            elif "-" not in split_tokens and "-" in current_segment:
                # If there is no ',', split by '-'
                sub_segments = TextTokenizer.split_segments_by_token(
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

    @staticmethod
    def split_segments(
        tokenized: Sequence[str], max_text_tokens_per_segment: int = 120, quick_streaming_tokens: int = 0
    ) -> list[list[str]]:
        return TextTokenizer.split_segments_by_token(
            tokenized,
            _PUNCTUATION_MARKS_TOKENS,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
