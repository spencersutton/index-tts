import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple, is_torch_flex_attn_available
from .configuration_opt import OPTConfig

"""PyTorch OPT model."""
if is_torch_flex_attn_available(): ...
logger = ...

class OPTLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = ...,
        position_ids: torch.LongTensor | None = ...,
    ):  # -> Tensor:

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

class OPTAttention(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx: int | None = ..., **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...

class OPTDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OPTConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        past_key_value: tuple[torch.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class OPTPreTrainedModel(PreTrainedModel):
    config: OPTConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

class OPTDecoder(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_decoder(self):  # -> OPTDecoder:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

class OPTForCausalLM(OPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> OPTDecoder:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...

class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: list[torch.FloatTensor] | Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: list[torch.FloatTensor] | Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

__all__ = [
    "OPTForCausalLM",
    "OPTForQuestionAnswering",
    "OPTForSequenceClassification",
    "OPTModel",
    "OPTPreTrainedModel",
]
