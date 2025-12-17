from dataclasses import dataclass

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple, is_torch_flex_attn_available
from .configuration_idefics import IdeficsConfig

"""PyTorch Idefics model."""
if is_torch_flex_attn_available(): ...
logger = ...

@dataclass
class IdeficsBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
class IdeficsCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: tuple[torch.FloatTensor] | None = ...

def expand_inputs_for_generation(
    input_ids, expand_size=..., is_encoder_decoder=..., attention_mask=..., encoder_outputs=..., **model_kwargs
):  # -> tuple[Any, dict[str, Any]]:
    ...
def freeze_model(model, module_exceptions=...): ...

class IdeficsDecoupledEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze: bool | None = ...,
        device=...,
        dtype=...,
        padding_idx=...,
        **kwargs,
    ) -> None: ...
    def forward(self, input_ids):  # -> Tensor:

        ...
    def extra_repr(self) -> str: ...

class IdeficsDecoupledLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = ...,
        bias: bool = ...,
        partially_freeze: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

class IdeficsRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class IdeficsEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=...) -> None: ...
    def forward(self, x, seq_len=...):  # -> tuple[Tensor | Any, Tensor | Any]:
        ...

def rotate_half(x):  # -> Tensor:

    ...
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=...):  # -> tuple[Any, Any]:

    ...

class IdeficsMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None: ...
    def forward(self, x):  # -> Any:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class IdeficsAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = ...,
        is_cross_attention: bool = ...,
        config: PretrainedConfig = ...,
        qk_layer_norms: bool = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: tuple[torch.Tensor] | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class IdeficsDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: IdeficsConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: tuple[torch.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class IdeficsGatedCrossAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: IdeficsConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        image_hidden_states: torch.Tensor | None = ...,
        image_attention_mask: torch.Tensor | None = ...,
        cross_attention_gate: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        past_key_value: tuple[torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class IdeficsPreTrainedModel(PreTrainedModel):
    config: IdeficsConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...

class IdeficsModel(IdeficsPreTrainedModel):
    def __init__(self, config: IdeficsConfig) -> None: ...
    def freeze_relevant_params(self, config=...):  # -> None:
        ...
    def freeze_text_layers(self, module_exceptions=...):  # -> None:
        ...
    def freeze_vision_layers(self, module_exceptions=...):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_encoder_embeddings: torch.FloatTensor | None = ...,
        perceiver_embeddings: torch.FloatTensor | None = ...,
        image_attention_mask: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | IdeficsBaseModelOutputWithPast: ...

class IdeficsForVisionText2Text(IdeficsPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config, vision_model=...) -> None: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> IdeficsModel:
        ...
    def tie_weights(self):  # -> None:

        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_encoder_embeddings: torch.FloatTensor | None = ...,
        perceiver_embeddings: torch.FloatTensor | None = ...,
        image_attention_mask: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | IdeficsCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=...,
        position_ids=...,
        inputs_embeds=...,
        past_key_values=...,
        cache_position=...,
        pixel_values=...,
        image_hidden_states=...,
        image_attention_mask=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = ["IdeficsForVisionText2Text", "IdeficsModel", "IdeficsPreTrainedModel"]
