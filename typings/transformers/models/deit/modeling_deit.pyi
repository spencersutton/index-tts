from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_deit import DeiTConfig

"""PyTorch DeiT model."""
logger = ...

class DeiTEmbeddings(nn.Module):
    def __init__(self, config: DeiTConfig, use_mask_token: bool = ...) -> None: ...
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.Tensor: ...

class DeiTPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

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

class DeiTSelfAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(
        self, hidden_states, head_mask: torch.Tensor | None = ..., output_attentions: bool = ...
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class DeiTSelfOutput(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class DeiTAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def prune_heads(self, heads: set[int]) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = ..., output_attentions: bool = ...
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class DeiTIntermediate(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class DeiTOutput(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class DeiTLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = ..., output_attentions: bool = ...
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: ...

class DeiTEncoder(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ) -> tuple | BaseModelOutput: ...

class DeiTPreTrainedModel(PreTrainedModel):
    config: DeiTConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class DeiTModel(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig, add_pooling_layer: bool = ..., use_mask_token: bool = ...) -> None: ...
    def get_input_embeddings(self) -> DeiTPatchEmbeddings: ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class DeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(self, hidden_states): ...

class DeiTForMaskedImageModeling(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        bool_masked_pos: torch.BoolTensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | MaskedImageModelingOutput: ...

class DeiTForImageClassification(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | ImageClassifierOutput: ...

@dataclass
class DeiTForImageClassificationWithTeacherOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    cls_logits: torch.FloatTensor | None = ...
    distillation_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | DeiTForImageClassificationWithTeacherOutput: ...

__all__ = [
    "DeiTForImageClassification",
    "DeiTForImageClassificationWithTeacher",
    "DeiTForMaskedImageModeling",
    "DeiTModel",
    "DeiTPreTrainedModel",
]
