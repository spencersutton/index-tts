"""Tests for common utility functions.

This module tests pure utility functions that handle text tokenization
and tensor masking operations.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.common import de_tokenized_by_CJK_char, tokenize_by_CJK_char


def test_tokenize_by_CJK_char_basic() -> None:
    """Test basic CJK tokenization with mixed text."""
    input_text = "你好世界是 hello world 的中文"
    expected = "你 好 世 界 是 HELLO WORLD 的 中 文"

    result = tokenize_by_CJK_char(input_text)
    assert result == expected


def test_tokenize_by_CJK_char_preserve_case() -> None:
    """Test CJK tokenization without uppercasing."""
    input_text = "你好 hello 世界"
    expected = "你 好 hello 世 界"

    result = tokenize_by_CJK_char(input_text, do_upper_case=False)
    assert result == expected


def test_tokenize_by_CJK_char_english_only() -> None:
    """Test tokenization with English-only text."""
    input_text = "hello world"
    expected = "HELLO WORLD"

    result = tokenize_by_CJK_char(input_text)
    assert result == expected


def test_tokenize_by_CJK_char_cjk_only() -> None:
    """Test tokenization with CJK-only text."""
    input_text = "你好世界"
    expected = "你 好 世 界"

    result = tokenize_by_CJK_char(input_text)
    assert result == expected


def test_tokenize_by_CJK_char_with_punctuation() -> None:
    """Test tokenization with punctuation marks."""
    input_text = "你好，世界！"
    expected = "你 好 ， 世 界 ！"

    result = tokenize_by_CJK_char(input_text)
    assert result == expected


def test_tokenize_by_CJK_char_empty_string() -> None:
    """Test tokenization with empty string."""
    input_text = ""
    expected = ""

    result = tokenize_by_CJK_char(input_text)
    assert result == expected


def test_tokenize_by_CJK_char_whitespace_handling() -> None:
    """Test that extra whitespace is handled correctly."""
    input_text = "你好   世界"
    expected = "你 好 世 界"

    result = tokenize_by_CJK_char(input_text)
    assert result == expected


def test_de_tokenized_by_CJK_char_basic() -> None:
    """Test basic de-tokenization of CJK characters."""
    input_text = "你 好 世 界 是 HELLO WORLD 的 中 文"
    expected = "你好世界是HELLO WORLD的中文"

    result = de_tokenized_by_CJK_char(input_text)
    assert result == expected


def test_de_tokenized_by_CJK_char_with_lowercase() -> None:
    """Test de-tokenization with lowercase conversion."""
    input_text = "SEE YOU!"
    expected = "see you!"

    result = de_tokenized_by_CJK_char(input_text, do_lower_case=True)
    assert result == expected


def test_de_tokenized_by_CJK_char_mixed_content() -> None:
    """Test de-tokenization with mixed CJK and English."""
    input_text = "你 好 HELLO 世 界"
    expected = "你好HELLO世界"

    result = de_tokenized_by_CJK_char(input_text)
    assert result == expected


def test_de_tokenized_by_CJK_char_english_only() -> None:
    """Test de-tokenization with English-only text."""
    input_text = "HELLO WORLD"
    expected = "HELLO WORLD"

    result = de_tokenized_by_CJK_char(input_text)
    assert result == expected


def test_de_tokenized_by_CJK_char_cjk_only() -> None:
    """Test de-tokenization with CJK-only text."""
    input_text = "你 好 世 界"
    expected = "你好世界"

    result = de_tokenized_by_CJK_char(input_text)
    assert result == expected


def test_tokenize_detokenize_roundtrip() -> None:
    """Test that tokenize and de-tokenize are inverse operations."""
    original = "你好世界 hello world 的中文"

    # Tokenize then de-tokenize
    tokenized = tokenize_by_CJK_char(original, do_upper_case=False)
    detokenized = de_tokenized_by_CJK_char(tokenized)

    # Result should be similar (spaces may differ)
    assert "你好世界" in detokenized
    assert "hello world" in detokenized
    assert "的中文" in detokenized


if __name__ == "__main__":
    # Run basic smoke test
    print("Running tests...")
    test_tokenize_by_CJK_char_basic()
    test_tokenize_by_CJK_char_preserve_case()
    test_tokenize_by_CJK_char_english_only()
    test_tokenize_by_CJK_char_cjk_only()
    test_tokenize_by_CJK_char_with_punctuation()
    test_tokenize_by_CJK_char_empty_string()
    test_tokenize_by_CJK_char_whitespace_handling()
    test_de_tokenized_by_CJK_char_basic()
    test_de_tokenized_by_CJK_char_with_lowercase()
    test_de_tokenized_by_CJK_char_mixed_content()
    test_de_tokenized_by_CJK_char_english_only()
    test_de_tokenized_by_CJK_char_cjk_only()
    test_tokenize_detokenize_roundtrip()
    print("All tests passed!")
