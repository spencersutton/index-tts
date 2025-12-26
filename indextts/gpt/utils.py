"""Utility functions for GPT model operations.

This module contains pure helper functions extracted from the main model
to improve maintainability and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


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
    # NOTE: This function must be compatible with torch.export/torch.compile.
    # Avoid Python control flow (loops/ifs) that depends on tensor data.
    #
    # Build a mask for all positions >= length for each batch item.
    # input_tokens: (B, T)
    # lengths:      (B,)
    seq_len = input_tokens.shape[-1]
    positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0)
    mask = positions >= lengths.to(device=input_tokens.device).unsqueeze(1)
    input_tokens.masked_fill_(mask, stop_token)
    return input_tokens
