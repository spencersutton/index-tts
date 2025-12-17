from dataclasses import dataclass

import torch
from torch import nn

from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available
from .configuration_mamba2 import Mamba2Config

"""PyTorch MAMBA2 model."""
logger = ...
if is_mamba_2_ssm_available(): ...
if is_causal_conv1d_available(): ...
is_fast_path_available = ...

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int): ...
def reshape_into_chunks(input_tensor, pad_size, chunk_size): ...
def segment_sum(input_tensor):  # -> Tensor:

    ...
def apply_mask_to_padding_states(hidden_states, attention_mask): ...

class Mamba2Cache:
    def __init__(
        self, config: Mamba2Config, batch_size: int, dtype: torch.dtype = ..., device: str | None = ...
    ) -> None: ...
    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = ...
    ) -> torch.Tensor: ...
    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):  # -> Tensor:
        ...
    def reset(self):  # -> None:
        ...

class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states, gate=...): ...

class Mamba2Mixer(nn.Module):
    def __init__(self, config: Mamba2Config, layer_idx: int) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Mamba2Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def torch_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Mamba2Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: Mamba2Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...

class Mamba2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...

class Mamba2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states,
        cache_params: Mamba2Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ): ...

class Mamba2PreTrainedModel(PreTrainedModel):
    config: Mamba2Config
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _is_stateful = ...

@dataclass
class Mamba2Output(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    cache_params: Mamba2Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
class Mamba2CausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    cache_params: Mamba2Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

class Mamba2Model(Mamba2PreTrainedModel):
    def __init__(self, config) -> None: ...
    def load_hook(self, state_dict, prefix, *args):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        cache_params: Mamba2Cache | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | Mamba2Output: ...

class Mamba2ForCausalLM(Mamba2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=...,
        use_cache=...,
        cache_params: Mamba2Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_params: Mamba2Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | Mamba2CausalLMOutput: ...

__all__ = ["Mamba2ForCausalLM", "Mamba2Model", "Mamba2PreTrainedModel"]
