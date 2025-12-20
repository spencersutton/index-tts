from functools import cached_property

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple
from .configuration_chameleon import ChameleonConfig, ChameleonVQVAEConfig

"""PyTorch Chameleon model."""
logger = ...

class ChameleonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class ChameleonRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...) -> None: ...
    @torch.no_grad()
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class ChameleonLinearScalingRotaryEmbedding(ChameleonRotaryEmbedding):
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class ChameleonDynamicNTKScalingRotaryEmbedding(ChameleonRotaryEmbedding):
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:

    ...
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=..., unsqueeze_dim=...):  # -> tuple[Any, Any]:

    ...

class ChameleonMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class ChameleonLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, *args, **kwargs) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
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

class ChameleonAttention(nn.Module):
    def __init__(self, config: ChameleonConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class ChameleonDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ChameleonConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class ChameleonSwinDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ChameleonConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class ChameleonVQVAEVectorQuantizer(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state: torch.Tensor):  # -> tuple[Any, Any | Tensor, Tensor]:
        ...

class ChameleonVQVAEEncoderConvDownsample(nn.Module):
    def __init__(self, in_channels) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class ChameleonVQVAEEncoderResnetBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels=..., conv_shortcut=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class ChameleonVQVAEEncoderAttnBlock(nn.Module):
    def __init__(self, in_channels) -> None: ...
    def forward(self, hidden_states): ...

class ChameleonVQVAEEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.LongTensor):  # -> Any:
        ...

class ChameleonImageVocabularyMapping:
    def __init__(self, vocab_map) -> None: ...
    @cached_property
    def val2name(self):  # -> dict[Any, Any]:
        ...
    @cached_property
    def image_tokens(self):  # -> list[Any]:
        ...
    @cached_property
    def bpe2img(self):  # -> dict[Any, int]:
        ...
    @cached_property
    def img2bpe(self):  # -> dict[int, Any]:
        ...
    @cached_property
    def bpe2img_search_tensors(self):  # -> tuple[Tensor, Tensor]:
        ...
    @cached_property
    def img2bpe_mapping_tensor(self):  # -> Tensor:
        ...
    def convert_img2bpe(self, img_batch: torch.Tensor) -> torch.Tensor: ...

class ChameleonPreTrainedModel(PreTrainedModel):
    config: ChameleonConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_param_buffer_assignment = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class ChameleonVQVAE(ChameleonPreTrainedModel):
    config: ChameleonVQVAEConfig
    _no_split_modules = ...
    def __init__(self, config: ChameleonVQVAEConfig) -> None: ...
    def encode(self, pixel_values: torch.LongTensor):  # -> tuple[Any, Any, Any]:
        ...

class ChameleonModel(ChameleonPreTrainedModel):
    def __init__(self, config: ChameleonConfig) -> None: ...
    def get_image_tokens(self, pixel_values: torch.FloatTensor):  # -> Tensor:

        ...
    def get_image_features(self, pixel_values: torch.FloatTensor):  # -> Any:

        ...
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):  # -> Any:

        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class ChameleonForConditionalGeneration(ChameleonPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> ChameleonModel:
        ...
    def get_image_tokens(self, pixel_values):  # -> Tensor:
        ...
    def get_image_features(self, pixel_values):  # -> Any:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=...,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = ["ChameleonForConditionalGeneration", "ChameleonModel", "ChameleonPreTrainedModel", "ChameleonVQVAE"]
