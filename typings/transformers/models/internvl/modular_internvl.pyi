from dataclasses import dataclass

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import can_return_tuple
from ..clip.modeling_clip import CLIPMLP
from ..janus.modeling_janus import JanusVisionAttention
from ..llama.modeling_llama import LlamaRMSNorm
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from .configuration_internvl import InternVLConfig, InternVLVisionConfig

logger = ...

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

class InternVLVisionRMSNorm(LlamaRMSNorm): ...

class InternVLVisionAttention(JanusVisionAttention):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):  # -> tuple[Any, Any] | tuple[Any, None]:
        ...

class InternVLVisionPreTrainedModel(PreTrainedModel):
    config: InternVLVisionConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

@dataclass
class InternVLVisionModelOutputWithPooling(BaseModelOutputWithPooling): ...

class InternVLVisionPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class InternVLVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.BoolTensor | None = ...) -> torch.Tensor: ...

class InternVLVisionMLP(CLIPMLP): ...

NORM2FN = ...

class InternVLVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = ...
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class InternVLVisionEncoder(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = ..., output_hidden_states: bool = ...
    ) -> tuple | BaseModelOutput: ...

class InternVLVisionModel(InternVLVisionPreTrainedModel):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> InternVLVisionPatchEmbeddings:
        ...
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | InternVLVisionModelOutputWithPooling: ...

class InternVLPreTrainedModel(LlavaPreTrainedModel): ...

INTERNVL_INPUTS_DOCSTRING = ...

class InternVLMultiModalProjector(nn.Module):
    def __init__(self, config: InternVLConfig) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

class InternVLModelOutputWithPast(LlavaModelOutputWithPast): ...

class InternVLModel(LlavaModel):
    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = ...):  # -> Tensor:

        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> Any:

        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        pixel_values: torch.FloatTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | InternVLModelOutputWithPast: ...

class InternVLCausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...

class InternVLForConditionalGeneration(LlavaForConditionalGeneration):
    def forward(**super_kwargs):  # -> None:

        ...

__all__ = [
    "InternVLForConditionalGeneration",
    "InternVLModel",
    "InternVLPreTrainedModel",
    "InternVLVisionModel",
    "InternVLVisionPreTrainedModel",
]
