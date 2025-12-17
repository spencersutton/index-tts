from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import can_return_tuple
from ...utils.generic import check_model_inputs
from ..esm.modeling_esm import (
    EsmAttention,
    EsmEmbeddings,
    EsmEncoder,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmPooler,
    EsmSelfAttention,
    EsmSelfOutput,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from .configuration_evolla import EvollaConfig, SaProtConfig

logger = ...

class EvollaSaProtEmbeddings(EsmEmbeddings):
    def __init__(self, config) -> None: ...

def rotate_half_esm(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb_esm(x, cos, sin): ...

class EvollaSaProtRotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None: ...
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

class EvollaSaProtSelfAttention(EsmSelfAttention, nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...

class EvollaSaProtSelfOutput(EsmSelfOutput): ...
class EvollaSaProtAttention(EsmAttention): ...
class EvollaSaProtIntermediate(EsmIntermediate): ...
class EvollaSaProtOutput(EsmOutput): ...
class EvollaSaProtLayer(EsmLayer): ...
class EvollaSaProtEncoder(EsmEncoder): ...
class EvollaSaProtPooler(EsmPooler): ...

class EvollaSaProtPreTrainedModel(PreTrainedModel):
    config: SaProtConfig
    _no_split_modules = ...
    _supports_flash_attn = ...

class EvollaSaProtProteinEncoder(EvollaSaProtPreTrainedModel):
    def __init__(self, config: SaProtConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: tuple[int], device: torch.device = ..., dtype: torch.float = ...
    ) -> Tensor: ...

class EvollaSequenceCompressorAttention(nn.Module):
    def __init__(self, dim, dim_head=..., heads=...) -> None: ...
    def forward(self, x, latents, mask):  # -> Any:

        ...

class EvollaFeedForward(nn.Module):
    def __init__(self, dim, mult=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class EvollaSequenceCompressorResampler(nn.Module):
    def __init__(self, config: EvollaConfig) -> None: ...
    def forward(self, embeds, mask):  # -> Any:
        ...

@dataclass
class EvollaProteinEncoderModelOutput(ModelOutput):
    sequence_compressor_output: torch.FloatTensor = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class EvollaProteinEncoder(nn.Module):
    def __init__(self, config: EvollaConfig) -> None: ...
    @can_return_tuple
    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, **kwargs
    ):  # -> EvollaProteinEncoderModelOutput:
        ...

class EvollaSequenceAlignerCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        protein_encoder_dim: int | None = ...,
        structure_encoder_dim: int | None = ...,
        msa_encoder_dim: int | None = ...,
    ) -> None: ...
    def cross_attention(
        self,
        query_states,
        protein_key_value_states,
        structure_key_value_states,
        msa_key_value_states,
        query_attn_mask,
        protein_kv_attn_mask,
        structure_kv_attn_mask,
        msa_kv_attn_mask,
    ):  # -> Any:

        ...
    def forward(
        self,
        query_states,
        protein_kv_states,
        structure_kv_states,
        msa_kv_states,
        query_attn_mask,
        protein_kv_attn_mask=...,
        structure_kv_attn_mask=...,
        msa_kv_attn_mask=...,
        protein_batch_mask=...,
        structure_batch_mask=...,
        msa_batch_mask=...,
        past_key_value=...,
    ): ...

class EvollaRMSNorm(LlamaRMSNorm): ...
class EvollaRotaryEmbedding(LlamaRotaryEmbedding): ...
class EvollaMLP(LlamaMLP): ...
class EvollaAttention(LlamaAttention): ...

class EvollaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: EvollaConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        protein_kv_states: torch.Tensor | None = ...,
        structure_kv_states: torch.Tensor | None = ...,
        msa_kv_states: torch.Tensor | None = ...,
        protein_batch_mask: torch.Tensor | None = ...,
        structure_batch_mask: torch.Tensor | None = ...,
        msa_batch_mask: torch.Tensor | None = ...,
        query_attn_mask: torch.Tensor | None = ...,
        **kwargs,
    ):  # -> Tensor:
        ...

class EvollaPreTrainedModel(LlamaPreTrainedModel):
    _supports_attention_backend = ...
    _no_split_modules = ...

class EvollaModel(EvollaPreTrainedModel):
    def __init__(self, config: EvollaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        protein_input_ids: torch.LongTensor | None = ...,
        protein_attention_mask: torch.Tensor | None = ...,
        structure_feats: torch.FloatTensor | None = ...,
        msa_feats: torch.FloatTensor | None = ...,
        structure_batch_mask: torch.Tensor | None = ...,
        msa_batch_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast: ...

class EvollaForProteinText2Text(EvollaPreTrainedModel, GenerationMixin):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        protein_input_ids: torch.LongTensor = ...,
        protein_attention_mask: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs,
    ):  # -> CausalLMOutputWithPast:

        ...

__all__ = ["EvollaForProteinText2Text", "EvollaModel", "EvollaPreTrainedModel"]
