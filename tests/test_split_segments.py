"""Tests for TextTokenizer.split_segments_by_token and split_segments.

These tests pin the exact observable behavior of the function so that
any refactoring can be validated without a live tokenizer.
"""

import warnings

import pytest

from indextts.utils.front import TextTokenizer

split = TextTokenizer.split_segments_by_token

PUNCT = [".", "!", "?", "▁.", "▁?", "▁..."]


# ---------------------------------------------------------------------------
# Basic / edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    assert split([], PUNCT, max_text_tokens_per_segment=10) == []


def test_single_token_no_split() -> None:
    assert split(["hello"], PUNCT, max_text_tokens_per_segment=10) == [["hello"]]


def test_tokens_below_max_no_split_token_hit() -> None:
    tokens = ["hello", "world"]
    assert split(tokens, PUNCT, max_text_tokens_per_segment=10) == [["hello", "world"]]


# ---------------------------------------------------------------------------
# Splitting at punctuation
# ---------------------------------------------------------------------------


def test_split_at_period() -> None:
    # Three tokens ending with "."
    tokens = ["hello", "world", "."]
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10)
    assert result == [["hello", "world", "."]]


def test_split_across_two_sentences() -> None:
    # Two full sentences
    tokens = ["one", "two", ".", "three", "four", "."]
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10)
    # Both sentences fit and are merged because combined length (6) <= max (10)
    assert result == [["one", "two", ".", "three", "four", "."]]


def test_split_across_two_sentences_too_large_to_merge() -> None:
    tokens = ["a", "b", "c", ".", "d", "e", "f", "."]
    # max=5 means each 4-token sentence fits individually but combined (8 > 5) won't merge,
    # and 8 > max/2=2.5 so no small-segment merge either
    result = split(tokens, PUNCT, max_text_tokens_per_segment=5)
    assert result == [["a", "b", "c", "."], ["d", "e", "f", "."]]


def test_no_split_when_segment_too_short() -> None:
    # Segment length == 2, the rule requires > 2 to split
    tokens = ["a", "!"]
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10)
    assert result == [["a", "!"]]


# ---------------------------------------------------------------------------
# Quote lookahead
# ---------------------------------------------------------------------------


def test_quote_lookahead_included_in_current_segment_and_next() -> None:
    """When the token after a split point is a quote, the quote is appended to
    the current (flushed) segment.  Because the for-loop does not skip the
    quote, it is also re-processed as the first token of the next segment."""
    tokens = ["a", "b", "!", "'", "c"]
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10)
    # quote appears at end of first segment AND start of second
    assert result == [["a", "b", "!", "'", "'", "c"]]  # merged because 5+2? wait let me recalculate

    # Actually: segments before merge = [["a", "b", "!", "'"], ["'", "c"]]
    # 4 + 2 = 6 <= 10, so they merge -> [["a", "b", "!", "'", "'", "c"]]


def test_quote_lookahead_splits_preserved_when_too_large() -> None:
    """Same as above but max is small enough to prevent merging."""
    tokens = ["a", "b", "!", "'", "c"]
    result = split(tokens, PUNCT, max_text_tokens_per_segment=4)
    # segments before merge = [["a", "b", "!", "'"], ["'", "c"]]
    # 4 + 2 = 6 > max=4, and 6 > max/2=2 -> no merge
    assert result == [["a", "b", "!", "'"], ["'", "c"]]


# ---------------------------------------------------------------------------
# Exceeding max length
# ---------------------------------------------------------------------------


def test_segment_exceeds_max_splits_by_size_with_warning() -> None:
    # 6 tokens, max=3 → two chunks of 3
    tokens = ["a", "b", "c", "d", "e", "f"]
    with pytest.warns(RuntimeWarning, match="exceeds limit"):
        result = split(tokens, PUNCT, max_text_tokens_per_segment=3)
    assert result == [["a", "b", "c"], ["d", "e", "f"]]


def test_segment_exceeds_max_uneven_splits() -> None:
    tokens = ["a", "b", "c", "d", "e"]
    with pytest.warns(RuntimeWarning, match="exceeds limit"):
        result = split(tokens, PUNCT, max_text_tokens_per_segment=3)
    assert result == [["a", "b", "c"], ["d", "e"]]


# ---------------------------------------------------------------------------
# Recursive comma split
# ---------------------------------------------------------------------------


def test_comma_triggers_recursive_split() -> None:
    """When split_tokens doesn't include ',' but the growing segment contains
    one, the function recursively splits by comma first."""
    # Using "." as the outer split token; comma appears inside a segment
    tokens = ["a", "b", ",", "c", "d", "."]
    # max=4: after processing up to "," the recursive comma-split fires
    # recursive(["a","b",","], [",","▁,"]) -> [["a","b","," ]] (len=3 > 2 → split)
    # then ["c","d","."] → split at "." → [["c","d","."]]
    # segments = [["a","b",","], ["c","d","."]]
    # 3+3=6 > max=4, and 6 > max/2=2 → no merge
    result = split(tokens, PUNCT, max_text_tokens_per_segment=4)
    assert result == [["a", "b", ","], ["c", "d", "."]]


# ---------------------------------------------------------------------------
# Recursive dash split
# ---------------------------------------------------------------------------


def test_dash_triggers_recursive_split() -> None:
    """When split_tokens doesn't include '-' and no comma is present, the
    function recursively splits by dash."""
    tokens = ["a", "b", "-", "c", "d", "."]
    # max=4: up to "-" → recursive([..], ["-"]) → split at "-" → [["a","b","-"]]
    # then ["c","d","."] → "." split → [["c","d","."]]
    # segments = [["a","b","-"], ["c","d","."]]
    # 3+3=6 > 4, no merge
    result = split(tokens, PUNCT, max_text_tokens_per_segment=4)
    assert result == [["a", "b", "-"], ["c", "d", "."]]


# ---------------------------------------------------------------------------
# Merge behaviour with quick_streaming_tokens
# ---------------------------------------------------------------------------


def test_merge_disabled_before_quick_streaming_threshold() -> None:
    """With quick_streaming_tokens set, short segments are kept separate until
    the cumulative token count exceeds the threshold."""
    # Two short sentences, total = 6 tokens
    tokens = ["a", "b", ".", "c", "d", "."]
    # max=10, quick_streaming_tokens=10 (threshold not reached until total > 10)
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10, quick_streaming_tokens=10)
    # segments before merge phase = [["a","b","."], ["c","d","."]]
    # total after seg1 = 3, after seg2 = 6 — both <= 10 (threshold)
    # combined 3+3=6 <= max, but total never exceeds quick_streaming_tokens=10
    # However, 6 <= max/2=5? No (6 > 5) → no small-segment merge
    assert result == [["a", "b", "."], ["c", "d", "."]]


def test_merge_enabled_after_quick_streaming_threshold() -> None:
    """Once cumulative tokens exceed quick_streaming_tokens, adjacent segments
    that fit within max are merged."""
    tokens = ["a", "b", ".", "c", "d", "."]
    # quick_streaming_tokens=4: total after seg2 = 6 > 4 → merge permitted
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10, quick_streaming_tokens=4)
    assert result == [["a", "b", ".", "c", "d", "."]]


def test_always_merge_when_combined_fits_in_half_max() -> None:
    """Segments are always merged when combined length <= max/2, regardless of
    quick_streaming_tokens."""
    tokens = ["a", ".", "b", "."]
    # Each segment is 2 tokens; combined = 4. max=10, max/2=5. 4 <= 5 → always merge.
    result = split(tokens, PUNCT, max_text_tokens_per_segment=10, quick_streaming_tokens=100)
    assert result == [["a", ".", "b", "."]]


# ---------------------------------------------------------------------------
# split_segments (public API wrapper)
# ---------------------------------------------------------------------------


def test_split_segments_uses_punctuation_tokens() -> None:
    tokens = ["hello", "world", "."]
    result = TextTokenizer.split_segments(tokens, max_text_tokens_per_segment=10)
    assert result == [["hello", "world", "."]]


def test_split_segments_empty() -> None:
    assert TextTokenizer.split_segments([], max_text_tokens_per_segment=10) == []
