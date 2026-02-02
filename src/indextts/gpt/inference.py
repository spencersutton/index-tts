from typing import Any, cast, override

import torch
import transformers
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from indextts.gpt.learned_pos_emb import LearnedPositionEmbeddings
from indextts.util import patch_call


class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    embeddings: nn.Embedding
    text_pos_embedding: LearnedPositionEmbeddings
    transformer: GPT2Model
    final_norm: nn.Module
    lm_head: nn.Sequential

    def __init__(
        self,
        config: GPT2Config,
        gpt: GPT2Model,
        text_pos_emb: LearnedPositionEmbeddings,
        embeddings: nn.Embedding,
        norm: nn.Module,
        linear: nn.Module,
    ) -> None:
        super().__init__(config)
        # Note: the argument named `text_pos_emb` here actually represents the mel position embedding
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)

        self.cached_mel_emb: Tensor | None = None

    @override
    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Tensor, "b t"],
        past_key_values: transformers.Cache | None = None,
        attention_mask: Int[Tensor, "b t"] | None = None,
        inputs_embeds: Tensor | None = None,
        cache_position: Tensor | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> dict[str, transformers.Cache | Tensor | bool | None]:
        token_type_ids = cast(Tensor | None, kwargs.get("token_type_ids"))  # usually None
        position_ids = kwargs.get("position_ids")
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

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
            "token_type_ids": token_type_ids,
        }

    @override
    def forward(
        self,
        input_ids: Int[Tensor, "b t"],
        past_key_values: transformers.Cache | None = None,
        attention_mask: Int[Tensor, "b t"] | None = None,
        token_type_ids: Int[Tensor, ""] | None = None,
        position_ids: Int[Tensor, "b t"] | None = None,
        head_mask: Float[Tensor, ""] | None = None,
        inputs_embeds: Float[Tensor, "b t d"] | None = None,
        encoder_hidden_states: Float[Tensor, ""] | None = None,
        encoder_attention_mask: Int[Tensor, ""] | None = None,
        labels: None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithCrossAttentions | tuple[Tensor, ...]:
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        assert self.cached_mel_emb is not None, "cached_mel_emb must be set before calling forward()"

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb += self.text_pos_embedding.__call__(text_emb.shape[1])
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0] // self.cached_mel_emb.shape[0], 0)
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            assert attention_mask is not None
            emb = self.embeddings(input_ids)
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
        hidden_states: Tensor = transformer_outputs[0]  # pyright: ignore

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits, *transformer_outputs[1:])

        assert not isinstance(transformer_outputs, tuple)
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
