from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...integrations import use_kernel_forward_from_hub
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, can_return_tuple
from .configuration_aimv2 import Aimv2Config, Aimv2TextConfig, Aimv2VisionConfig

@dataclass
class Aimv2Output(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

@use_kernel_forward_from_hub("RMSNorm")
class Aimv2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class Aimv2MLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class Aimv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=..., temperature=..., device=..., dtype=...
    ) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class Aimv2TextEmbeddings(nn.Module):
    def __init__(self, config: Aimv2TextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

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

class Aimv2Attention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = ..., **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Aimv2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Aimv2Encoder(nn.Module):
    def __init__(self, config: Aimv2Config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutput: ...

class Aimv2AttentionPoolingHead(nn.Module):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Aimv2PreTrainedModel(PreTrainedModel):
    config: Aimv2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...

class Aimv2VisionModel(Aimv2PreTrainedModel):
    config: Aimv2VisionConfig
    main_input_name = ...
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    def forward(
        self,
        pixel_values,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

class Aimv2TextModel(Aimv2PreTrainedModel):
    main_input_name = ...
    def __init__(self, config: Aimv2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

class Aimv2Model(Aimv2PreTrainedModel):
    config: Aimv2Config
    _no_split_modules = ...
    def __init__(self, config: Aimv2Config) -> None: ...
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> torch.FloatTensor: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> Aimv2Output: ...

__all__ = ["Aimv2Model", "Aimv2PreTrainedModel", "Aimv2TextModel", "Aimv2VisionModel"]
