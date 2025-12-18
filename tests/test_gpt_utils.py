"""Tests for GPT utility functions.

This module tests the pure helper functions extracted from the GPT model
to ensure they work correctly in isolation.
"""

import sys
from pathlib import Path

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.gpt.utils import (
    build_aligned_inputs_and_targets,
    build_hf_gpt_transformer,
    set_token_padding,
)


def test_build_aligned_inputs_and_targets_basic() -> None:
    """Test basic functionality of build_aligned_inputs_and_targets."""
    input_ids = torch.tensor([[1, 2, 3]])
    start_token = 0
    stop_token = 4

    inp, tar = build_aligned_inputs_and_targets(input_ids, start_token, stop_token)

    # Input should have start_token prepended
    assert inp.shape == (1, 4)
    assert inp[0, 0].item() == start_token
    assert torch.equal(inp[0, 1:], input_ids[0])

    # Target should have stop_token appended
    assert tar.shape == (1, 4)
    assert torch.equal(tar[0, :-1], input_ids[0])
    assert tar[0, -1].item() == stop_token


def test_build_aligned_inputs_and_targets_batch() -> None:
    """Test build_aligned_inputs_and_targets with batch processing."""
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    start_token = 0
    stop_token = 99

    inp, tar = build_aligned_inputs_and_targets(input_ids, start_token, stop_token)

    # Check batch dimension is preserved
    assert inp.shape == (2, 4)
    assert tar.shape == (2, 4)

    # Check first batch element
    assert inp[0, 0].item() == start_token
    assert torch.equal(inp[0, 1:], input_ids[0])
    assert tar[0, -1].item() == stop_token

    # Check second batch element
    assert inp[1, 0].item() == start_token
    assert torch.equal(inp[1, 1:], input_ids[1])
    assert tar[1, -1].item() == stop_token


def test_build_aligned_inputs_and_targets_empty() -> None:
    """Test build_aligned_inputs_and_targets with empty sequence."""
    input_ids = torch.tensor([[]])
    start_token = 0
    stop_token = 1

    inp, tar = build_aligned_inputs_and_targets(input_ids, start_token, stop_token)

    # Should have exactly one element (the start/stop token)
    assert inp.shape == (1, 1)
    assert tar.shape == (1, 1)
    assert inp[0, 0].item() == start_token
    assert tar[0, 0].item() == stop_token


def test_set_token_padding_basic() -> None:
    """Test basic functionality of set_token_padding."""
    tokens = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    lengths = torch.tensor([3, 2])
    stop_token = 99

    result = set_token_padding(tokens, lengths, stop_token)

    # First sequence: first 3 elements unchanged, rest should be stop_token
    assert torch.equal(result[0, :3], torch.tensor([1, 2, 3]))
    assert torch.all(result[0, 3:] == stop_token)

    # Second sequence: first 2 elements unchanged, rest should be stop_token
    assert torch.equal(result[1, :2], torch.tensor([1, 2]))
    assert torch.all(result[1, 2:] == stop_token)


def test_set_token_padding_no_padding_needed() -> None:
    """Test set_token_padding when sequences are already full length."""
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    lengths = torch.tensor([5])
    stop_token = 99

    result = set_token_padding(tokens, lengths, stop_token)

    # No padding should be added since length equals sequence length
    assert torch.equal(result, tokens)


def test_set_token_padding_all_padding() -> None:
    """Test set_token_padding when entire sequence should be padding."""
    tokens = torch.tensor([[0, 0, 0, 0, 0]])
    lengths = torch.tensor([0])
    stop_token = 99

    result = set_token_padding(tokens, lengths, stop_token)

    # All tokens should be replaced with stop_token
    assert torch.all(result == stop_token)


def test_set_token_padding_partial() -> None:
    """Test set_token_padding with varying lengths in a batch."""
    tokens = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 0, 0, 0],
        [8, 0, 0, 0, 0],
    ])
    lengths = torch.tensor([5, 2, 1])
    stop_token = 999

    result = set_token_padding(tokens, lengths, stop_token)

    # First sequence: no padding
    assert torch.equal(result[0], tokens[0])

    # Second sequence: padding after position 2
    assert torch.equal(result[1, :2], torch.tensor([6, 7]))
    assert torch.all(result[1, 2:] == stop_token)

    # Third sequence: padding after position 1
    assert result[2, 0].item() == 8
    assert torch.all(result[2, 1:] == stop_token)


def test_set_token_padding_inplace_modification() -> None:
    """Test that set_token_padding modifies tensor in-place."""
    tokens = torch.tensor([[1, 2, 0, 0]])
    lengths = torch.tensor([2])
    stop_token = 99

    # Store original id to verify it's the same tensor
    original_id = id(tokens)

    result = set_token_padding(tokens, lengths, stop_token)

    # Result should be the same tensor object (in-place modification)
    assert id(result) == original_id
    assert torch.all(result[0, 2:] == stop_token)


def test_build_hf_gpt_transformer_basic() -> None:
    """Test build_hf_gpt_transformer creates valid components."""
    layers = 2
    model_dim = 128
    heads = 4
    max_mel_seq_len = 100
    max_text_seq_len = 50

    gpt, mel_pos_emb, text_pos_emb, _, _ = build_hf_gpt_transformer(
        layers, model_dim, heads, max_mel_seq_len, max_text_seq_len
    )

    # Check GPT model was created
    assert gpt is not None
    assert len(gpt.h) == layers  # Number of transformer blocks

    # Check position embeddings were created
    assert mel_pos_emb is not None
    assert text_pos_emb is not None

    # Verify GPT config
    assert gpt.config.n_embd == model_dim
    assert gpt.config.n_layer == layers
    assert gpt.config.n_head == heads
    assert gpt.config.n_positions == max_mel_seq_len + max_text_seq_len
    assert gpt.config.gradient_checkpointing is True
    assert gpt.config.use_cache is False


def test_build_hf_gpt_transformer_custom_null_embedding() -> None:
    """Test that build_hf_gpt_transformer uses NullPositionEmbedding."""
    from indextts.gpt.utils import NullPositionEmbedding

    layers = 1
    model_dim = 64
    heads = 2
    max_mel_seq_len = 10
    max_text_seq_len = 10

    gpt, _, _, _, _ = build_hf_gpt_transformer(
        layers, model_dim, heads, max_mel_seq_len, max_text_seq_len
    )

    # Check that wpe is a NullPositionEmbedding
    assert isinstance(gpt.wpe, NullPositionEmbedding)

    # Verify wte (token embeddings) has been deleted
    # Note: This is hard to test directly since the attribute is deleted


def test_build_hf_gpt_transformer_different_seq_lengths() -> None:
    """Test build_hf_gpt_transformer with different sequence lengths."""
    layers = 3
    model_dim = 256
    heads = 8
    max_mel_seq_len = 500
    max_text_seq_len = 200

    gpt, mel_pos_emb, text_pos_emb, _, _ = build_hf_gpt_transformer(
        layers, model_dim, heads, max_mel_seq_len, max_text_seq_len
    )

    # Total positions should be sum of mel and text
    assert gpt.config.n_positions == max_mel_seq_len + max_text_seq_len
    assert gpt.config.n_ctx == max_mel_seq_len + max_text_seq_len


if __name__ == "__main__":
    # Run basic smoke test
    print("Running tests...")
    test_build_aligned_inputs_and_targets_basic()
    test_build_aligned_inputs_and_targets_batch()
    test_build_aligned_inputs_and_targets_empty()
    test_set_token_padding_basic()
    test_set_token_padding_no_padding_needed()
    test_set_token_padding_all_padding()
    test_set_token_padding_partial()
    test_set_token_padding_inplace_modification()
    test_build_hf_gpt_transformer_basic()
    test_build_hf_gpt_transformer_custom_null_embedding()
    test_build_hf_gpt_transformer_different_seq_lengths()
    print("All tests passed!")
