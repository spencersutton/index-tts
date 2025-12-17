import torch
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...models.modernbert.modeling_modernbert import ModernBertPreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple
from ...utils.generic import check_model_inputs

logger = ...

class ModernBertDecoderConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_activation=...,
        max_position_embeddings=...,
        initializer_range=...,
        initializer_cutoff_factor=...,
        norm_eps=...,
        norm_bias=...,
        pad_token_id=...,
        eos_token_id=...,
        bos_token_id=...,
        cls_token_id=...,
        sep_token_id=...,
        global_rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        embedding_dropout=...,
        mlp_bias=...,
        mlp_dropout=...,
        decoder_bias=...,
        classifier_dropout=...,
        classifier_bias=...,
        classifier_activation=...,
        use_cache=...,
        local_attention=...,
        global_attn_every_n_layers=...,
        local_rope_theta=...,
        layer_types=...,
        **kwargs,
    ) -> None: ...

def eager_attention_forward(
    module: ModernBertDecoderAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = ...,
    scaling: float | None = ...,
    sliding_window: int | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class ModernBertDecoderAttention(nn.Module):
    def __init__(self, config: ModernBertDecoderConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class ModernBertDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ModernBertDecoderConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class ModernBertDecoderPreTrainedModel(ModernBertPreTrainedModel):
    config: ModernBertDecoderConfig
    base_model_prefix = ...
    _skip_keys_device_placement = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_gradient_checkpointing = ...
    _can_compile_fullgraph = ...
    _supports_attention_backend = ...
    _can_record_outputs = ...

class ModernBertDecoderModel(ModernBertDecoderPreTrainedModel):
    def __init__(self, config: ModernBertDecoderConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPast: ...

class ModernBertDecoderForCausalLM(ModernBertDecoderPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: ModernBertDecoderConfig) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...

class ModernBertDecoderForSequenceClassification(ModernBertDecoderPreTrainedModel):
    def __init__(self, config: ModernBertDecoderConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs,
    ) -> tuple | SequenceClassifierOutputWithPast: ...

__all__ = [
    "ModernBertDecoderConfig",
    "ModernBertDecoderForCausalLM",
    "ModernBertDecoderForSequenceClassification",
    "ModernBertDecoderModel",
    "ModernBertDecoderPreTrainedModel",
]
