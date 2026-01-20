import os
from collections.abc import Callable, Generator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self, TypeVar

import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch import nn
from transformers.utils import is_torchao_available as is_torchao_available

from .activations import get_activation as get_activation
from .configuration_utils import PretrainedConfig as PretrainedConfig
from .dynamic_module_utils import custom_object_save as custom_object_save
from .generation import CompileConfig as CompileConfig
from .generation import GenerationConfig as GenerationConfig
from .integrations import PeftAdapterMixin as PeftAdapterMixin
from .integrations import deepspeed_config as deepspeed_config
from .integrations import is_deepspeed_zero3_enabled as is_deepspeed_zero3_enabled
from .integrations.accelerate import find_tied_parameters as find_tied_parameters
from .integrations.accelerate import init_empty_weights as init_empty_weights
from .integrations.flash_attention import flash_attention_forward as flash_attention_forward
from .integrations.flex_attention import flex_attention_forward as flex_attention_forward
from .integrations.sdpa_attention import sdpa_attention_forward as sdpa_attention_forward
from .integrations.tensor_parallel import SUPPORTED_TP_STYLES as SUPPORTED_TP_STYLES
from .integrations.tensor_parallel import repack_weights as repack_weights
from .integrations.tensor_parallel import replace_state_dict_local_with_dtensor as replace_state_dict_local_with_dtensor
from .integrations.tensor_parallel import shard_and_distribute_module as shard_and_distribute_module
from .integrations.tensor_parallel import verify_tp_plan as verify_tp_plan
from .loss.loss_utils import LOSS_MAPPING as LOSS_MAPPING
from .pytorch_utils import Conv1D as Conv1D
from .pytorch_utils import apply_chunking_to_forward as apply_chunking_to_forward
from .pytorch_utils import find_pruneable_heads_and_indices as find_pruneable_heads_and_indices
from .pytorch_utils import id_tensor_storage as id_tensor_storage
from .pytorch_utils import prune_conv1d_layer as prune_conv1d_layer
from .pytorch_utils import prune_layer as prune_layer
from .pytorch_utils import prune_linear_layer as prune_linear_layer
from .quantizers import AutoHfQuantizer as AutoHfQuantizer
from .quantizers import HfQuantizer as HfQuantizer
from .quantizers.quantizers_utils import get_module_from_name as get_module_from_name
from .safetensors_conversion import auto_conversion as auto_conversion
from .utils import ADAPTER_SAFE_WEIGHTS_NAME as ADAPTER_SAFE_WEIGHTS_NAME
from .utils import ADAPTER_WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from .utils import CONFIG_NAME as CONFIG_NAME
from .utils import DUMMY_INPUTS as DUMMY_INPUTS
from .utils import FLAX_WEIGHTS_NAME as FLAX_WEIGHTS_NAME
from .utils import SAFE_WEIGHTS_INDEX_NAME as SAFE_WEIGHTS_INDEX_NAME
from .utils import SAFE_WEIGHTS_NAME as SAFE_WEIGHTS_NAME
from .utils import TF2_WEIGHTS_NAME as TF2_WEIGHTS_NAME
from .utils import TF_WEIGHTS_NAME as TF_WEIGHTS_NAME
from .utils import WEIGHTS_INDEX_NAME as WEIGHTS_INDEX_NAME
from .utils import WEIGHTS_NAME as WEIGHTS_NAME
from .utils import ContextManagers as ContextManagers
from .utils import ModelOutput as ModelOutput
from .utils import PushToHubMixin as PushToHubMixin
from .utils import cached_file as cached_file
from .utils import check_torch_load_is_safe as check_torch_load_is_safe
from .utils import copy_func as copy_func
from .utils import download_url as download_url
from .utils import extract_commit_hash as extract_commit_hash
from .utils import find_adapter_config_file as find_adapter_config_file
from .utils import has_file as has_file
from .utils import is_accelerate_available as is_accelerate_available
from .utils import is_bitsandbytes_available as is_bitsandbytes_available
from .utils import is_flash_attn_2_available as is_flash_attn_2_available
from .utils import is_kernels_available as is_kernels_available
from .utils import is_offline_mode as is_offline_mode
from .utils import is_optimum_available as is_optimum_available
from .utils import is_peft_available as is_peft_available
from .utils import is_remote_url as is_remote_url
from .utils import is_safetensors_available as is_safetensors_available
from .utils import is_torch_flex_attn_available as is_torch_flex_attn_available
from .utils import is_torch_greater_or_equal as is_torch_greater_or_equal
from .utils import is_torch_mlu_available as is_torch_mlu_available
from .utils import is_torch_npu_available as is_torch_npu_available
from .utils import is_torch_sdpa_available as is_torch_sdpa_available
from .utils import is_torch_xla_available as is_torch_xla_available
from .utils import is_torch_xpu_available as is_torch_xpu_available
from .utils import logging as logging
from .utils import replace_return_docstrings as replace_return_docstrings
from .utils import strtobool as strtobool
from .utils.hub import create_and_tag_model_card as create_and_tag_model_card
from .utils.hub import get_checkpoint_shard_files as get_checkpoint_shard_files
from .utils.import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
from .utils.import_utils import is_huggingface_hub_greater_or_equal as is_huggingface_hub_greater_or_equal
from .utils.import_utils import is_sagemaker_mp_enabled as is_sagemaker_mp_enabled
from .utils.import_utils import is_torch_fx_proxy as is_torch_fx_proxy
from .utils.import_utils import is_torchdynamo_compiling as is_torchdynamo_compiling
from .utils.quantization_config import BitsAndBytesConfig as BitsAndBytesConfig
from .utils.quantization_config import QuantizationMethod as QuantizationMethod

XLA_USE_BF16: Incomplete
XLA_DOWNCAST_BF16: Incomplete
accelerate_version: Incomplete
logger: Incomplete

def is_fsdp_enabled(): ...
def is_local_dist_rank_0(): ...

IS_SAGEMAKER_MP_POST_1_10: Incomplete
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound=PreTrainedModel)
TORCH_INIT_FUNCTIONS: Incomplete
VLMS: Incomplete

@contextmanager
def no_init_weights() -> Generator[None]: ...
@contextmanager
def set_quantized_state() -> Generator[None]: ...
@contextmanager
def set_zero3_state() -> Generator[None]: ...
def restore_default_torch_dtype(func): ...
def get_torch_context_manager_or_global_device(): ...
def get_parameter_device(parameter: nn.Module | ModuleUtilsMixin): ...
def get_parameter_dtype(parameter: nn.Module | ModuleUtilsMixin): ...
def get_state_dict_dtype(state_dict): ...
def load_sharded_checkpoint(model, folder, strict: bool = True, prefer_safe: bool = True): ...

str_to_torch_dtype: Incomplete

def load_state_dict(
    checkpoint_file: str | os.PathLike,
    is_quantized: bool = False,
    map_location: str | torch.device | None = "cpu",
    weights_only: bool = True,
): ...
def set_initialized_submodules(model, state_dict_keys): ...

class PipelineParallel(Enum):
    inputs: 0
    outputs: 1

class ModuleUtilsMixin:
    def add_memory_hooks(self) -> None: ...
    def reset_memory_hooks_state(self) -> None: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor: ...
    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None): ...
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor: ...
    def get_head_mask(
        self, head_mask: Tensor | None, num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor: ...
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int: ...
    warnings_issued: Incomplete
    def estimate_tokens(self, input_dict: dict[str, torch.Tensor | Any]) -> int: ...
    def floating_point_ops(self, input_dict: dict[str, torch.Tensor | Any], exclude_embeddings: bool = True) -> int: ...

class PreTrainedModel(nn.Module, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
    config_class: Incomplete
    base_model_prefix: str
    main_input_name: str
    model_tags: Incomplete
    is_parallelizable: bool
    supports_gradient_checkpointing: bool
    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]: ...
    @property
    def framework(self) -> str: ...
    config: Incomplete
    loss_type: Incomplete
    name_or_path: Incomplete
    warnings_issued: Incomplete
    generation_config: Incomplete
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs) -> None: ...
    def post_init(self) -> None: ...
    def dequantize(self): ...
    def add_model_tags(self, tags: list[str] | str) -> None: ...
    @property
    def base_model(self) -> nn.Module: ...
    @classmethod
    def can_generate(cls) -> bool: ...
    def enable_input_require_grads(self) -> None: ...
    def disable_input_require_grads(self) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value: nn.Module): ...
    def get_output_embeddings(self) -> nn.Module: ...
    def initialize_weights(self): ...
    def tie_weights(self) -> None: ...
    vocab_size: Incomplete
    def resize_token_embeddings(
        self, new_num_tokens: int | None = None, pad_to_multiple_of: int | None = None, mean_resizing: bool = True
    ) -> nn.Embedding: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int): ...
    def get_position_embeddings(self) -> nn.Embedding | tuple[nn.Embedding]: ...
    def init_weights(self) -> None: ...
    def prune_heads(self, heads_to_prune: dict[int, list[int]]): ...
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None: ...
    def gradient_checkpointing_disable(self) -> None: ...
    @property
    def is_gradient_checkpointing(self) -> bool: ...
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool = True,
        state_dict: dict | None = None,
        save_function: Callable = ...,
        push_to_hub: bool = False,
        max_shard_size: int | str = "5GB",
        safe_serialization: bool = True,
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        **kwargs,
    ): ...
    def push_to_hub(self, *args, **kwargs): ...
    def get_memory_footprint(self, return_buffers: bool = True): ...
    def cuda(self, *args, **kwargs) -> Self: ...
    def to(self, *args, **kwargs) -> Self: ...
    def half(self, *args) -> Self: ...
    def float(self, *args) -> Self: ...
    @classmethod
    def get_init_context(cls, is_quantized: bool, _is_ds_init_called: bool): ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs,
    ) -> Self: ...
    def retrieve_modules_from_names(self, names, add_prefix: bool = False, remove_prefix: bool = False): ...
    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoModel") -> None: ...
    def to_bettertransformer(self) -> PreTrainedModel: ...
    def reverse_bettertransformer(self): ...
    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask) -> None: ...
    @property
    def supports_tp_plan(self): ...
    @property
    def tp_size(self): ...
    @property
    def supports_pp_plan(self): ...
    @property
    def loss_function(self): ...
    @loss_function.setter
    def loss_function(self, value) -> None: ...
    def get_compiled_call(self, compile_config: CompileConfig | None) -> Callable: ...
    @classmethod
    def is_backend_compatible(cls): ...
    def get_parameter_or_buffer(self, target: str): ...

class PoolerStartLogits(nn.Module):
    dense: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: torch.FloatTensor | None = None
    ) -> torch.FloatTensor: ...

class PoolerEndLogits(nn.Module):
    dense_0: Incomplete
    activation: Incomplete
    LayerNorm: Incomplete
    dense_1: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        p_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor: ...

class PoolerAnswerClass(nn.Module):
    dense_0: Incomplete
    activation: Incomplete
    dense_1: Incomplete
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        cls_index: torch.LongTensor | None = None,
    ) -> torch.FloatTensor: ...

@dataclass
class SquadHeadOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_top_log_probs: torch.FloatTensor | None = ...
    start_top_index: torch.LongTensor | None = ...
    end_top_log_probs: torch.FloatTensor | None = ...
    end_top_index: torch.LongTensor | None = ...
    cls_logits: torch.FloatTensor | None = ...
    def __post_init__(self) -> None: ...

class SQuADHead(nn.Module):
    start_n_top: Incomplete
    end_n_top: Incomplete
    start_logits: Incomplete
    end_logits: Incomplete
    answer_class: Incomplete
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_positions: torch.LongTensor | None = None,
        end_positions: torch.LongTensor | None = None,
        cls_index: torch.LongTensor | None = None,
        is_impossible: torch.LongTensor | None = None,
        p_mask: torch.FloatTensor | None = None,
        return_dict: bool = False,
    ) -> SquadHeadOutput | tuple[torch.FloatTensor]: ...

class SequenceSummary(nn.Module):
    summary_type: Incomplete
    summary: Incomplete
    activation: Callable
    first_dropout: Incomplete
    last_dropout: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: torch.LongTensor | None = None
    ) -> torch.FloatTensor: ...

def unwrap_model(model: nn.Module, recursive: bool = False) -> nn.Module: ...
def expand_device_map(device_map, param_names): ...
def is_accelerator_device(device: str | int | torch.device) -> bool: ...
def caching_allocator_warmup(model: PreTrainedModel, expanded_device_map: dict, hf_quantizer: HfQuantizer | None): ...
def get_disk_only_shard_files(device_map, weight_map): ...

class AttentionInterface(MutableMapping):
    def __init__(self) -> None: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    @classmethod
    def register(cls, key: str, value: Callable): ...
    def valid_keys(self) -> list[str]: ...

ALL_ATTENTION_FUNCTIONS: AttentionInterface
