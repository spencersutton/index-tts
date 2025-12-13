from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import Cache, Conv1D, GPT2Config, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model

from .attention import Attention


class GPT2AccelAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        max_positions = cast(int, config.max_position_embeddings)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = cast(int, config.hidden_size)
        self.num_heads = cast(int, config.num_attention_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            msg = (
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
            raise ValueError(msg)

        self.scale_attn_weights = config.scale_attn_weights

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(cast(float, config.attn_pdrop))
        self.resid_dropout = nn.Dropout(cast(float, config.resid_pdrop))

        scale = (self.head_dim**-0.5) if self.scale_attn_weights else 1.0
        self.accel_attn = Attention(self.num_heads, self.head_dim, scale, self.num_heads)

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor | None, ...]:
        if encoder_hidden_states is not None:
            msg = "Cross attention not supported in accel mode"
            raise NotImplementedError(msg)

        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        # [B, T, H*D] -> [B, H, T, D]
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # flatten to [B*T, H, D]
        bsz, num_heads, seq_len, head_dim = query.shape
        q_flat = query.transpose(1, 2).contiguous().view(-1, num_heads, head_dim)
        k_flat = key.transpose(1, 2).contiguous().view(-1, num_heads, head_dim)
        v_flat = value.transpose(1, 2).contiguous().view(-1, num_heads, head_dim)

        # ensure fp16
        if q_flat.device.type == "cuda" and q_flat.dtype != torch.float16:
            orig_dtype = q_flat.dtype
            q_flat = q_flat.to(torch.float16)
            k_flat = k_flat.to(torch.float16)
            v_flat = v_flat.to(torch.float16)
        else:
            orig_dtype = q_flat.dtype

        o_flat = self.accel_attn(q_flat, k_flat, v_flat)  # [B*T, H, D]

        if o_flat.dtype != orig_dtype:
            o_flat = o_flat.to(orig_dtype)

        # Reshape back: [B*T, H, D] -> [B, H, T, D]
        attn_output = o_flat.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, None)
        if output_attentions:
            outputs += (None,)

        return outputs

    def _split_heads(self, tensor: Tensor, num_heads: int, head_dim: int) -> Tensor:
        new_shape = (*tensor.size()[:-1], num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor: Tensor, num_heads: int, head_dim: int) -> Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = (*tensor.size()[:-2], num_heads * head_dim)
        return tensor.view(new_shape)


class GPT2AccelBlock(GPT2Block):
    def __init__(self, config: PretrainedConfig, layer_idx: int | None = None) -> None:
        super().__init__(config, layer_idx)
        self.attn = GPT2AccelAttention(config, layer_idx)


class GPT2AccelModel(GPT2Model):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.h = nn.ModuleList([GPT2AccelBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | Cache | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[torch.FloatTensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

            for block in self.h:
                assert isinstance(block, GPT2AccelBlock)
                hidden_states = block(hidden_states)[0]

            hidden_states = self.ln_f(hidden_states)

            if return_dict:
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                )
            return (hidden_states,)
        return super().forward(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
