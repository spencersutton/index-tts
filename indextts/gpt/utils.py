"""Utility functions for GPT model operations.

This module contains pure helper functions extracted from the main model
to improve maintainability and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def set_token_padding(input_tokens: Tensor, stop_token: int) -> Tensor:
    mask = (input_tokens == stop_token).cummax(dim=-1).values
    return input_tokens.masked_fill_(mask, stop_token)
