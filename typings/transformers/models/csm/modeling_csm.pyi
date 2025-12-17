from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.utils.generic import check_model_inputs

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, can_return_tuple
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

@use_kernel_forward_from_hub("RMSNorm")
class CsmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class CsmRotaryEmbedding(nn.Module):
    def __init__(self, config: CsmConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class CsmMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

def rotate_half(x):  # -> Tensor:

    ...
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=..., unsqueeze_dim=...):  # -> tuple[Any, Any]:

    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class CsmAttention(nn.Module):
    def __init__(self, config: CsmConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class CsmDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CsmConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]: ...

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

class CsmDepthDecoderModel(CsmPreTrainedModel):
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

class CsmDepthDecoderForCausalLM(CsmPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config) -> None: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> CsmDepthDecoderModel:
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

class CsmBackboneModelEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids):  # -> Any:
        ...

class CsmBackboneModel(CsmPreTrainedModel):
    def __init__(self, config) -> None: ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast: ...

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
