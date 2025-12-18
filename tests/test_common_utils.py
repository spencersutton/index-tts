"""Tests for common utility functions.

This module tests pure utility functions that handle text tokenization
and tensor masking operations.
"""

import sys
from pathlib import Path

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.common import (
    de_tokenized_by_CJK_char,
    make_pad_mask,
    tokenize_by_CJK_char,
)


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


def test_make_pad_mask_basic() -> None:
    """Test basic padding mask creation."""
    lengths = torch.tensor([5, 3, 2])
    expected = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, True, True],
        [False, False, True, True, True],
    ])

    result = make_pad_mask(lengths)
    assert torch.equal(result, expected)


def test_make_pad_mask_with_max_len() -> None:
    """Test padding mask with explicit max length."""
    lengths = torch.tensor([3, 2])
    max_len = 5
    expected = torch.tensor([
        [False, False, False, True, True],
        [False, False, True, True, True],
    ])

    result = make_pad_mask(lengths, max_len=max_len)
    assert torch.equal(result, expected)


def test_make_pad_mask_single_element() -> None:
    """Test padding mask with single batch element."""
    lengths = torch.tensor([4])
    expected = torch.tensor([[False, False, False, False]])

    result = make_pad_mask(lengths)
    assert torch.equal(result, expected)


def test_make_pad_mask_all_same_length() -> None:
    """Test padding mask when all sequences have same length."""
    lengths = torch.tensor([5, 5, 5])
    expected = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
    ])

    result = make_pad_mask(lengths)
    assert torch.equal(result, expected)


def test_make_pad_mask_zero_length() -> None:
    """Test padding mask with zero-length sequence."""
    lengths = torch.tensor([3, 0, 2])
    expected = torch.tensor([
        [False, False, False],
        [True, True, True],
        [False, False, True],
    ])

    result = make_pad_mask(lengths)
    assert torch.equal(result, expected)


def test_make_pad_mask_device_consistency() -> None:
    """Test that output mask is on the same device as input."""
    lengths = torch.tensor([3, 2])

    result = make_pad_mask(lengths)

    # Both should be on same device (CPU in this test environment)
    assert result.device == lengths.device


def test_make_pad_mask_dtype() -> None:
    """Test that output mask has correct dtype (bool)."""
    lengths = torch.tensor([3, 2])

    result = make_pad_mask(lengths)

    assert result.dtype == torch.bool


def test_make_pad_mask_shape() -> None:
    """Test that output mask has correct shape."""
    lengths = torch.tensor([5, 3, 7])
    max_len = 10

    result = make_pad_mask(lengths, max_len=max_len)

    # Shape should be (batch_size, max_len)
    assert result.shape == (3, 10)


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
    test_make_pad_mask_basic()
    test_make_pad_mask_with_max_len()
    test_make_pad_mask_single_element()
    test_make_pad_mask_all_same_length()
    test_make_pad_mask_zero_length()
    test_make_pad_mask_device_consistency()
    test_make_pad_mask_dtype()
    test_make_pad_mask_shape()
    print("All tests passed!")
