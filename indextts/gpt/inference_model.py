"""GPT-2 inference model for TTS generation.

This module contains the GPT2InferenceModel class which wraps a GPT-2 model
for autoregressive mel token generation during inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import torch
from torch import Tensor, nn
from transformers import (
    Cache,
    GenerationMixin,
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.util import patch_call


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
        return torch.zeros(
            (input.shape[0], input.shape[1], self.embedding_dim),
            device=input.device,
            dtype=input.dtype,
        )

    @patch_call(forward)
    def __call__(self) -> None: ...


class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    """GPT-2 wrapper for TTS inference with mel token generation.

    This model wraps a GPT-2 model and adds TTS-specific functionality:
    - Custom position embeddings for text and mel tokens
    - Cached mel embeddings for efficient autoregressive generation
    - Optional model parallelism across GPUs
    """

    if TYPE_CHECKING:
        lm_head: nn.Sequential[nn.LayerNorm | nn.Linear]

    text_pos_embedding: LearnedPositionEmbeddings
    transformer: GPT2Model
    kv_cache: bool
    cached_mel_emb: Tensor | None
    device_map: dict[int, int] | None
    model_parallel: bool

    def __init__(
        self,
        config: GPT2Config,
        gpt: GPT2Model,
        text_pos_emb: LearnedPositionEmbeddings,
        embeddings: nn.Embedding,
        norm: nn.LayerNorm,
        linear: nn.Linear,
        kv_cache: bool = False,
    ) -> None:
        """Initialize the inference model.

        Args:
            config: GPT-2 configuration.
            gpt: The underlying GPT-2 model.
            text_pos_emb: Position embeddings for mel tokens (despite the name).
            embeddings: Token embeddings for mel tokens.
            norm: Layer normalization before the output head.
            linear: Linear projection to mel vocabulary.
            kv_cache: Whether to use key-value caching for faster generation.
        """
        super().__init__(config)
        # Note: the argument named `text_pos_emb` here actually represents the mel position embedding
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    @override
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def store_mel_emb(self, mel_emb: torch.Tensor) -> None:
        """Store mel embeddings for use during generation.

        These embeddings represent the conditioning context (speaker + text)
        and are prepended to the generated mel tokens.

        Args:
            mel_emb: The mel embeddings to cache, shape (batch, seq, dim).
        """
        self.cached_mel_emb = mel_emb

    @override
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Tensor,
    ) -> dict[str, Any]:
        inputs_embeds = kwargs.get("inputs_embeds")  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1].unsqueeze(-1)

        position_ids = kwargs.get("position_ids")

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": inputs_embeds,
        }

    @override
    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithCrossAttentions | tuple[Tensor, ...]:
        """Forward pass for mel token generation.

        Args:
            input_ids: Input token IDs, shape (batch, seq).
            past_key_values: Cached key-value pairs for efficient generation.
            attention_mask: Attention mask, shape (batch, seq).
            token_type_ids: Unused, kept for API compatibility.
            position_ids: Position IDs for the tokens.
            head_mask: Mask for attention heads.
            inputs_embeds: Not supported, must be None.
            encoder_hidden_states: Cross-attention hidden states.
            encoder_attention_mask: Cross-attention mask.
            labels: Not supported (training not implemented).
            use_cache: Whether to return cached key-value pairs.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return a ModelOutput object.

        Returns:
            CausalLMOutputWithCrossAttentions containing logits and optional
            cached values, or a tuple if return_dict is False.
        """
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb += self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0] // self.cached_mel_emb.shape[0], 0)
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            assert attention_mask is not None
            emb += self.text_pos_embedding.get_fixed_embedding(attention_mask.shape[1] - mel_len)

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        assert not isinstance(transformer_outputs, tuple)
        hidden_states = cast(Tensor, transformer_outputs[0])

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head[1].weight.device)

        lm_logits = cast(torch.FloatTensor, self.lm_head(hidden_states))

        if not return_dict:
            return (lm_logits, *transformer_outputs[1:])

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @patch_call(forward)
    def __call__(self) -> None: ...
