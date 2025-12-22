"""Pure audio utility functions for IndexTTS2.

This module contains stateless utility functions for audio processing
that don't depend on any model state or global configuration.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

# Default sample rate for output audio
OUTPUT_SR = 22050

# Default sample rate for semantic model
SEMANTIC_SR = 16000

# Maximum audio length in seconds
MAX_LEN = 15


def generate_silence_interval(
    wavs: Sequence[Tensor],
    interval_silence: int = 200,
    sample_rate: int = OUTPUT_SR,
) -> Tensor:
    """Generate a silence tensor to be inserted between audio segments.

    Args:
        wavs: A sequence of audio tensors, used to determine the channel size.
        interval_silence: Duration of silence in milliseconds.
        sample_rate: Sample rate for the silence tensor.

    Returns:
        A tensor of zeros representing silence, with the same number of channels
        as the first tensor in `wavs`.

    Raises:
        AssertionError: If `interval_silence` is not positive or `wavs` is empty.
    """
    assert interval_silence > 0, "interval_silence must be greater than 0"
    assert len(wavs) > 0, "wavs list must not be empty"

    # get channel_size
    channel_size = wavs[0].size(0)
    # get silence tensor
    sil_dur = int(sample_rate * interval_silence / 1000.0)
    ref = wavs[0]
    return torch.zeros(channel_size, sil_dur, device=ref.device, dtype=ref.dtype)


def insert_interval_silence(
    wavs: Sequence[Tensor],
    interval_silence: int = 200,
    sample_rate: int = OUTPUT_SR,
) -> list[Tensor]:
    """Insert silences between generated audio segments.

    Args:
        wavs: A sequence of audio tensors.
        interval_silence: Duration of silence in milliseconds to insert between segments.
        sample_rate: Sample rate for the silence tensors.

    Returns:
        A list of tensors with silence tensors inserted between each pair of
        original audio segments. Returns a copy of the input list if empty
        or if interval_silence is non-positive.
    """
    if not wavs or interval_silence <= 0:
        return list(wavs)

    # get channel_size
    channel_size = wavs[0].size(0)
    # get silence tensor
    sil_dur = int(sample_rate * interval_silence / 1000.0)
    ref = wavs[0]
    sil_tensor = torch.zeros(channel_size, sil_dur, device=ref.device, dtype=ref.dtype)

    wavs_list: list[Tensor] = []
    for i, wav in enumerate(wavs):
        wavs_list.append(wav)
        if i < len(wavs) - 1:
            wavs_list.append(sil_tensor)

    return wavs_list


def find_most_similar_cosine(query_vector: Tensor, matrix: Tensor) -> Tensor:
    """Find the index of the most similar vector in a matrix using cosine similarity.

    Args:
        query_vector: A 1D or 2D query tensor.
        matrix: A 2D tensor where each row is a candidate vector.

    Returns:
        The index (as a 0-d tensor) of the most similar row in the matrix.
    """
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = torch.cosine_similarity(query_vector, matrix, dim=1)
    return torch.argmax(similarities)
