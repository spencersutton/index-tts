"""Emotion vector utilities for IndexTTS2.

This module contains pure functions for emotion vector processing
that don't depend on any model state.
"""

from __future__ import annotations

from collections.abc import Sequence

# Emotion labels in order (for reference)
EMOTION_LABELS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")

# Default bias factors for each emotion
# These de-emphasize emotions that can cause strange results
DEFAULT_EMO_BIAS = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)

# Maximum allowed sum for emotion vectors
MAX_EMO_SUM = 0.8


def normalize_emo_vec(
    emo_vector: Sequence[float],
    apply_bias: bool = True,
    emo_bias: Sequence[float] = DEFAULT_EMO_BIAS,
    max_sum: float = MAX_EMO_SUM,
) -> list[float]:
    """Normalize an emotion vector by applying optional bias factors and scaling.

    This function applies predefined bias factors to de-emphasize certain emotions
    that can cause strange synthesis results, then scales the vector so the sum
    doesn't exceed a maximum threshold.

    Args:
        emo_vector: A sequence of emotion intensity values, typically in the order:
            [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
        apply_bias: Whether to apply predefined bias factors to de-emphasize
            certain emotions. Defaults to True.
        emo_bias: Custom bias factors to apply. Defaults to the standard bias.
        max_sum: Maximum allowed sum for the emotion vector. Defaults to 0.8.

    Returns:
        The normalized emotion vector, possibly biased and scaled so that the sum
        does not exceed `max_sum`.

    Examples:
        >>> normalize_emo_vec([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        [0.46875, 0.2625, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]

        >>> normalize_emo_vec([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], apply_bias=False)
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    """
    result: list[float]

    # apply biased emotion factors for better user experience,
    # by de-emphasizing emotions that can cause strange results
    if apply_bias:
        result = [vec * bias for vec, bias in zip(emo_vector, emo_bias, strict=False)]
    else:
        result = list(emo_vector)

    # the total emotion sum must not exceed max_sum
    emo_sum = sum(result)
    if emo_sum > max_sum:
        scale_factor = max_sum / emo_sum
        result = [vec * scale_factor for vec in result]

    return result


def scale_emo_vector(emo_vector: Sequence[float], alpha: float) -> list[float]:
    """Scale an emotion vector by an alpha factor, clamping to [0, 1].

    Args:
        emo_vector: The emotion vector to scale.
        alpha: Scale factor, clamped to [0, 1].

    Returns:
        The scaled emotion vector, truncated to 4 decimal places.
    """
    scale = max(0.0, min(1.0, alpha))
    if scale == 1.0:
        return list(emo_vector)
    return [int(x * scale * 10000) / 10000 for x in emo_vector]
