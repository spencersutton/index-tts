"""Utility functions for GPT model operations.

This module contains pure helper functions extracted from the main model
to improve maintainability and testability.
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model

from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings


class NullPositionEmbedding(nn.Embedding):
    """A position embedding that always returns zeros.

    Used to replace the built-in position embeddings in GPT-2 when we want
    to use custom position embeddings instead.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(1, dim)
        del self.weight

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Return zero embeddings for the given input shape."""
        return torch.zeros((input.shape[0], input.shape[1], self.embedding_dim))


def build_aligned_inputs_and_targets(
    input: Tensor,  # noqa: A002
    start_token: int,
    stop_token: int,
) -> tuple[Tensor, Tensor]:
    """Build aligned inputs and targets by padding with start/stop tokens.

    Args:
        input: Input tensor to be padded.
        start_token: Token ID to prepend to the input.
        stop_token: Token ID to append to the target.

    Returns:
        A tuple of (input_padded, target_padded) where:
        - input_padded has start_token prepended
        - target_padded has stop_token appended

    Example:
        >>> input_ids = torch.tensor([[1, 2, 3]])
        >>> inp, tar = build_aligned_inputs_and_targets(input_ids, start_token=0, stop_token=4)
        >>> inp
        tensor([[0, 1, 2, 3]])
        >>> tar
        tensor([[1, 2, 3, 4]])
    """
    inp = F.pad(input, (1, 0), value=start_token)
    tar = F.pad(input, (0, 1), value=stop_token)
    return inp, tar


def set_token_padding(
    input_tokens: Tensor,
    lengths: Tensor,
    stop_token: int,
) -> Tensor:
    """Set padding tokens in a batch of sequences.

    Given tokens derived from a padded sequence and the actual lengths of each
    batch element, reformats the tokens with stop_token in place of padding.

    Args:
        input_tokens: Token tensor of shape (batch, seq_len).
        lengths: Actual length of each sequence in the batch (batch,).
        stop_token: Token ID to use for padding.

    Returns:
        Modified token tensor with padding replaced by stop_token.

    Example:
        >>> tokens = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> lengths = torch.tensor([3, 2])
        >>> result = set_token_padding(tokens, lengths, stop_token=99)
        >>> result
        tensor([[ 1,  2,  3, 99, 99],
                [ 1,  2, 99, 99, 99]])
    """
    for b in range(len(lengths)):
        actual_end = lengths[b]
        if actual_end < input_tokens.shape[-1]:
            input_tokens[b, actual_end:] = stop_token
    return input_tokens


def build_hf_gpt_transformer(
    layers: int,
    model_dim: int,
    heads: int,
    max_mel_seq_len: int,
    max_text_seq_len: int,
) -> tuple[GPT2Model, LearnedPositionEmbeddings, LearnedPositionEmbeddings, None, None]:
    """Build a GPT-2 transformer using the HuggingFace library.

    Args:
        layers: Number of transformer layers.
        model_dim: Model dimension (embedding size).
        heads: Number of attention heads.
        max_mel_seq_len: Maximum mel sequence length.
        max_text_seq_len: Maximum text sequence length.

    Returns:
        A tuple of:
        - GPT2Model: The transformer model
        - LearnedPositionEmbeddings: Mel position embeddings
        - LearnedPositionEmbeddings: Text position embeddings
        - None: Placeholder for compatibility
        - None: Placeholder for compatibility
    """
    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=True,
        use_cache=False,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = NullPositionEmbedding(model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )
