from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils.generic import check_model_inputs

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, can_return_tuple
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    TransformersKwargs,
)
from .configuration_csm import CsmConfig, CsmDepthDecoderConfig
from .generation_csm import CsmGenerationMixin

logger = ...

@dataclass
class CsmOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor = ...
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    depth_decoder_loss: torch.FloatTensor | None = ...
    depth_decoder_logits: torch.FloatTensor = ...
    depth_decoder_past_key_values: tuple[tuple[torch.FloatTensor]] | None = ...
    depth_decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    depth_decoder_attentions: tuple[torch.FloatTensor, ...] | None = ...
    backbone_loss: torch.FloatTensor | None = ...

class CsmRMSNorm(LlamaRMSNorm): ...
class CsmRotaryEmbedding(LlamaRotaryEmbedding): ...
class CsmMLP(LlamaMLP): ...
class CsmAttention(LlamaAttention): ...
class CsmDecoderLayer(LlamaDecoderLayer): ...

class CsmPreTrainedModel(PreTrainedModel):
    config: CsmConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class CsmDepthDecoderModel(LlamaModel, CsmPreTrainedModel):
    config: CsmDepthDecoderConfig
    def __init__(self, config) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        backbone_last_hidden_state: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class CsmCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size) -> None: ...
    def forward(self, hidden_states, cache_position=...):  # -> Tensor:
        ...

class CsmDepthDecoderForCausalLM(LlamaForCausalLM, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config) -> None: ...
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        backbone_last_hidden_state: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...

class CsmBackboneModelEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids):  # -> Any:
        ...

class CsmBackboneModel(LlamaModel):
    def __init__(self, config) -> None: ...
    @check_model_inputs
    def forward(self, **super_kwargs):  # -> BaseModelOutputWithPast:

        ...

class CsmForConditionalGeneration(CsmPreTrainedModel, CsmGenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> CsmBackboneModelEmbeddings:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # -> tuple[Any | Self, Any] | Self:
        ...
    def save_pretrained(self, *args, **kwargs):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        input_values: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        input_values_cutoffs: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CsmOutputWithPast: ...

__all__ = [
    "CsmBackboneModel",
    "CsmDepthDecoderForCausalLM",
    "CsmDepthDecoderModel",
    "CsmForConditionalGeneration",
    "CsmPreTrainedModel",
]
