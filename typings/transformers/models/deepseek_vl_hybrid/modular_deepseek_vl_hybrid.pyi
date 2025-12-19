import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...image_processing_utils_fast import BatchFeature, DefaultFastImageProcessorKwargs
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    can_return_tuple,
    filter_out_non_signature_kwargs,
    is_torchvision_v2_available,
)
from ..auto import AutoConfig
from ..deepseek_vl.configuration_deepseek_vl import DeepseekVLConfig
from ..deepseek_vl.image_processing_deepseek_vl import DeepseekVLImageProcessor
from ..deepseek_vl.image_processing_deepseek_vl_fast import DeepseekVLImageProcessorFast
from ..deepseek_vl.modeling_deepseek_vl import (
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
    DeepseekVLPreTrainedModel,
)
from ..deepseek_vl.processing_deepseek_vl import DeepseekVLProcessor, DeepseekVLProcessorKwargs
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..sam.modeling_sam import SamLayerNorm, SamVisionNeck

if is_torchvision_v2_available(): ...
logger = ...
DEEPSEEK_VL_COMMON_CUSTOM_ARGS = ...

class DeepseekVLHybridConfig(DeepseekVLConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config: AutoConfig = ...,
        vision_config: AutoConfig = ...,
        high_res_vision_config: AutoConfig = ...,
        image_token_id: int = ...,
        **kwargs,
    ) -> None: ...

class DeepseekVLHybridBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast): ...
class DeepseekVLHybridCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast): ...
class DeepseekVLHybridLayerNorm(SamLayerNorm): ...

class DeepseekVLSamVisionNeck(SamVisionNeck):
    def __init__(self, config) -> None: ...

class DeepseekVLSamVisionProj(nn.Module):
    def __init__(self, config, output_size: int = ...) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class DeepseekVLHybridAligner(nn.Module):
    def __init__(self, config: DeepseekVLHybridConfig) -> None: ...
    def forward(self, vision_encodings: torch.Tensor, high_res_vision_encodings: torch.Tensor) -> torch.Tensor: ...

class DeepseekVLHybridPreTrainedModel(DeepseekVLPreTrainedModel): ...

class DeepseekVLHybridModel(DeepseekVLModel):
    def __init__(self, config) -> None: ...
    def get_low_res_image_features(self, pixel_values):  # -> Any:
        ...
    def get_high_res_image_features(self, pixel_values):  # -> Any:
        ...
    def get_image_features(self, pixel_values, high_res_pixel_values):  # -> Any:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        pixel_values: torch.FloatTensor = ...,
        high_res_pixel_values: torch.FloatTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ):  # -> DeepseekVLHybridBaseModelOutputWithPast:
        ...

class DeepseekVLHybridForConditionalGeneration(DeepseekVLForConditionalGeneration):
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        pixel_values: torch.FloatTensor = ...,
        high_res_pixel_values: torch.FloatTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> DeepseekVLHybridCausalLMOutputWithPast:

        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        high_res_pixel_values=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

class DeepseekVLHybridImageProcessor(DeepseekVLImageProcessor):
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        high_res_size: dict[str, int] | None = ...,
        min_size: int = ...,
        resample: PILImageResampling = ...,
        high_res_resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        high_res_image_mean: float | list[float] | None = ...,
        high_res_image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        **kwargs,
    ) -> None: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        high_res_size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        high_res_resample: PILImageResampling = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        high_res_image_mean: float | list[float] | None = ...,
        high_res_image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        do_convert_rgb: bool | None = ...,
    ):  # -> BatchFeature:

        ...

class DeepseekVLHybridFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    min_size: int
    high_res_size: dict
    high_res_resample: PILImageResampling
    high_res_image_mean: list[float]
    high_res_image_std: list[float]

class DeepseekVLHybridImageProcessorFast(DeepseekVLImageProcessorFast):
    high_res_image_mean = ...
    high_res_image_std = ...
    high_res_size = ...
    high_res_resample = ...
    def __init__(self, **kwargs: Unpack[DeepseekVLHybridFastImageProcessorKwargs]) -> None: ...

class DeepseekVLHybridProcessorKwargs(DeepseekVLProcessorKwargs): ...

class DeepseekVLHybridProcessor(DeepseekVLProcessor):
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = ...,
        images: ImageInput = ...,
        **kwargs: Unpack[DeepseekVLHybridProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = [
    "DeepseekVLHybridConfig",
    "DeepseekVLHybridForConditionalGeneration",
    "DeepseekVLHybridImageProcessor",
    "DeepseekVLHybridImageProcessorFast",
    "DeepseekVLHybridModel",
    "DeepseekVLHybridPreTrainedModel",
    "DeepseekVLHybridProcessor",
]
