"""Tests for GPT utility functions.

These tests cover small, pure helpers in `indextts.gpt.utils`.
"""

import sys
from pathlib import Path

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.gpt.utils import set_token_padding


def test_set_token_padding_inferrs_lengths_from_stop_token() -> None:
    """When lengths is omitted, padding is inferred from the first stop token."""
    stop_token = 7
    tokens = torch.tensor([
        [1, 2, stop_token, 3, 4],  # after stop -> forced to stop
        [5, 6, 8, 9, 10],  # no stop -> unchanged
    ])

    out = set_token_padding(tokens.clone(), stop_token=stop_token)

    assert torch.equal(out[0], torch.tensor([1, 2, stop_token, stop_token, stop_token]))
    assert torch.equal(out[1], tokens[1])


if __name__ == "__main__":
    # Run basic smoke test
    print("Running tests...")
    test_set_token_padding_inferrs_lengths_from_stop_token()
    print("All tests passed!")
