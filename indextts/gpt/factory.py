"""Factory functions for building GPT-2 transformer models.

This module contains factory functions for constructing the GPT-2 backbone
used in the TTS model.
"""

from __future__ import annotations

from transformers import GPT2Config, GPT2Model

from indextts.gpt.inference_model import NullPositionEmbedding
from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings


def build_hf_gpt_transformer(
    layers: int,
    model_dim: int,
    heads: int,
    max_mel_seq_len: int,
    max_text_seq_len: int,
) -> tuple[GPT2Model, LearnedPositionEmbeddings, LearnedPositionEmbeddings, None, None]:
    """Build a GPT-2 transformer model using HuggingFace's implementation.

    This creates a GPT-2 model with custom position embeddings suitable for
    TTS applications. The built-in position and token embeddings are replaced
    with custom learned embeddings.

    Args:
        layers: Number of transformer layers.
        model_dim: Hidden dimension size.
        heads: Number of attention heads.
        max_mel_seq_len: Maximum sequence length for mel tokens.
        max_text_seq_len: Maximum sequence length for text tokens.

    Returns:
        A tuple containing:
        - gpt: The GPT-2 model with modified embeddings.
        - mel_pos_embedding: Learned position embeddings for mel tokens.
        - text_pos_embedding: Learned position embeddings for text tokens.
        - None: Placeholder for mel layer position embeddings (unused).
        - None: Placeholder for text layer position embeddings (unused).
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
