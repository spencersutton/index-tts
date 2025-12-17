import torch
from torch import nn

from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import is_flash_attn_available
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel
from .configuration_bark import BarkCoarseConfig, BarkConfig, BarkFineConfig, BarkSemanticConfig, BarkSubModelConfig
from .generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)

"""PyTorch BARK model."""
if is_flash_attn_available(): ...
logger = ...

class BarkSelfAttention(nn.Module):
    def __init__(self, config, is_causal=..., layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        past_key_values=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any]:
        ...

class BarkSelfFlashAttention2(BarkSelfAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        past_key_values=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, None]:
        ...

BARK_ATTENTION_CLASSES = ...

class BarkMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class BarkBlock(GradientCheckpointingLayer):
    def __init__(self, config, is_causal=..., layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states,
        past_key_values=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class BarkPreTrainedModel(PreTrainedModel):
    config: BarkConfig
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    def __init__(self, *inputs, **kwargs) -> None: ...
    @property
    def device(self) -> torch.device: ...

class BarkCausalModel(BarkPreTrainedModel, GenerationMixin):
    config: BarkSubModelConfig
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=...,
        input_embeds=...,
        past_key_values=...,
        position_ids=...,
        use_cache=...,
        cache_position=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: tuple[torch.FloatTensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        input_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithPast: ...

class BarkSemanticModel(BarkCausalModel):
    base_model_prefix = ...
    config: BarkSemanticConfig
    def generate(
        self,
        input_ids: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = ...,
        history_prompt: dict[str, torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> torch.LongTensor: ...

class BarkCoarseModel(BarkCausalModel):
    base_model_prefix = ...
    config: BarkCoarseConfig
    def preprocess_histories(
        self,
        max_coarse_history: int,
        semantic_to_coarse_ratio: int,
        batch_size: int,
        semantic_generation_config: int,
        codebook_size: int,
        history_prompt: dict[str, torch.Tensor] | None = ...,
    ):  # -> tuple[Tensor, Tensor]:

        ...
    def generate(
        self,
        semantic_output: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = ...,
        coarse_generation_config: BarkCoarseGenerationConfig = ...,
        codebook_size: int = ...,
        history_prompt: dict[str, torch.Tensor] | None = ...,
        return_output_lengths: bool | None = ...,
        **kwargs,
    ) -> torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor]: ...

class BarkFineModel(BarkPreTrainedModel):
    base_model_prefix = ...
    config: BarkFineConfig
    main_input_name = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> ModuleList:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_output_embeddings(self):  # -> ModuleList:
        ...
    def set_output_embeddings(self, new_output_embeddings):  # -> None:
        ...
    def resize_token_embeddings(
        self, new_num_tokens: int | None = ..., pad_to_multiple_of: int | None = ..., mean_resizing: bool = ...
    ) -> nn.Embedding: ...
    def tie_weights(self):  # -> None:

        ...
    def forward(
        self,
        codebook_idx: int,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        input_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MaskedLMOutput: ...
    @torch.no_grad()
    def generate(
        self,
        coarse_output: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = ...,
        coarse_generation_config: BarkCoarseGenerationConfig = ...,
        fine_generation_config: BarkFineGenerationConfig = ...,
        codebook_size: int = ...,
        history_prompt: dict[str, torch.Tensor] | None = ...,
        **kwargs,
    ) -> torch.LongTensor: ...

class BarkModel(BarkPreTrainedModel):
    config: BarkConfig
    def __init__(self, config) -> None: ...
    @classmethod
    def can_generate(cls) -> bool: ...
    @property
    def device(self) -> torch.device: ...
    def enable_cpu_offload(self, accelerator_id: int | None = ..., **kwargs):  # -> None:

        ...
    def codec_decode(self, fine_output, output_lengths=...):  # -> list[Any] | Any:

        ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = ...,
        history_prompt: dict[str, torch.Tensor] | None = ...,
        return_output_lengths: bool | None = ...,
        **kwargs,
    ) -> torch.LongTensor: ...

__all__ = [
    "BarkCausalModel",
    "BarkCoarseModel",
    "BarkFineModel",
    "BarkModel",
    "BarkPreTrainedModel",
    "BarkSemanticModel",
]
