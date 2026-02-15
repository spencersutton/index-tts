from indextts.utils.front import TextTokenizer


def test_split_segments_empty_input() -> None:
    assert TextTokenizer.split_segments([], max_text_tokens_per_segment=10) == []


def test_split_segments_respects_apostrophe_lookahead_without_duplication() -> None:
    tokens = ["HELLO", ".", "'", "WORLD", "!"]
    result = TextTokenizer.split_segments(tokens, max_text_tokens_per_segment=4, quick_streaming_tokens=999)
    flattened = [t for seg in result for t in seg]
    assert flattened == tokens
    assert flattened.count("'") == 1


def test_split_segments_chunks_when_over_max_length() -> None:
    tokens = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    result = TextTokenizer.split_segments(tokens, max_text_tokens_per_segment=3, quick_streaming_tokens=999)
    assert result == [["t1", "t2", "t3"], ["t4"], ["t5", "t6", "t7"]]


def test_split_segments_fallback_comma_split() -> None:
    tokens = ["A", "X", ",", "B", "."]
    result = TextTokenizer.split_segments(tokens, max_text_tokens_per_segment=4, quick_streaming_tokens=999)
    assert result == [["A", "X", ","], ["B", "."]]
