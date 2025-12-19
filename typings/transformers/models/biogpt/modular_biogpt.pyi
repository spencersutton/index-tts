import torch

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, is_torch_flex_attn_available
from ..bart.modeling_bart import BartAttention, BartDecoderLayer, BartScaledWordEmbedding
from ..opt.modeling_opt import OPTLearnedPositionalEmbedding
from .configuration_biogpt import BioGptConfig

"""PyTorch BioGPT model."""
if is_torch_flex_attn_available(): ...

class BioGptLearnedPositionalEmbedding(OPTLearnedPositionalEmbedding):
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = ...,
        position_ids: torch.LongTensor | None = ...,
    ):  # -> None:

        ...

class BioGptScaledWordEmbedding(BartScaledWordEmbedding): ...
class BioGptAttention(BartAttention): ...

class BioGptDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BioGptConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class BioGptPreTrainedModel(PreTrainedModel):
    config: BioGptConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

class BioGptModel(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

class BioGptForCausalLM(BioGptPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

class BioGptForTokenClassification(BioGptPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

class BioGptForSequenceClassification(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...
    def get_input_embeddings(self):  # -> BioGptScaledWordEmbedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

__all__ = [
    "BioGptForCausalLM",
    "BioGptForSequenceClassification",
    "BioGptForTokenClassification",
    "BioGptModel",
    "BioGptPreTrainedModel",
]
