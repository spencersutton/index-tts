import torch
from torch import nn

from indextts.util import patch_call

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_gpt2 import GPT2Config

"""PyTorch OpenAI GPT-2 model."""
logger = ...

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path): ...
def eager_attention_forward(
    module, query, key, value, attention_mask, head_mask=..., **kwargs
):  # -> tuple[Tensor, Any]:
    ...

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, is_cross_attention: bool = ..., layer_idx: int | None = ...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: tuple[torch.Tensor] | None,
        past_key_value: Cache | None = ...,
        cache_position: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(self, hidden_states: tuple[torch.Tensor] | None) -> torch.Tensor: ...

class GPT2Block(GradientCheckpointingLayer):
    def __init__(self, config: GPT2Config, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.Tensor] | torch.Tensor | None,
        past_key_value: Cache | None = ...,
        cache_position: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs: object,
    ) -> tuple[torch.Tensor] | (tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None): ...

class GPT2SequenceSummary(nn.Module):
    def __init__(self, config: GPT2Config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, cls_index: torch.Tensor | None = ...) -> torch.Tensor: ...

class GPT2PreTrainedModel(PreTrainedModel):
    config: GPT2Config
    load_tf_weights = ...
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_attention_backend = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs: object, **kwargs: object) -> None: ...

class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: torch.Tensor | None = ...
    mc_loss: torch.Tensor | None = ...
    logits: torch.Tensor | None = ...
    mc_logits: torch.Tensor | None = ...
    past_key_values: tuple[tuple[torch.Tensor]] | None = ...
    hidden_states: tuple[torch.Tensor] | None = ...
    attentions: tuple[torch.Tensor] | None = ...

PARALLELIZE_DOCSTRING = ...
DEPARALLELIZE_DOCSTRING = ...

class GPT2Model(GPT2PreTrainedModel):
    h: nn.ModuleList = ...
    first_device: str = ...
    last_device: str = ...
    wpe: nn.Embedding = ...
    wte: nn.Embedding = ...
    ln_f: nn.LayerNorm = ...

    _supports_param_buffer_assignment = ...
    def __init__(self, config: GPT2Config) -> None: ...
    def parallelize(self, device_map: dict[int, int] | None = ...) -> None: ...
    def deparallelize(self) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | Cache | None = ...,
        cache_position: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def parallelize(self, device_map: dict | None = ...) -> None: ...
    def deparallelize(self) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        cache_position: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def parallelize(self, device_map=...):  # -> None:
        ...
    def deparallelize(self):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        cache_position: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        mc_token_ids: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        mc_labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | GPT2DoubleHeadsModelOutput: ...

class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...

class GPT2ForTokenClassification(GPT2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "GPT2LMHeadModel",
    "GPT2Model",
    "GPT2PreTrainedModel",
    "load_tf_weights_in_gpt2",
]
