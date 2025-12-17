import torch

from ...modeling_utils import PreTrainedModel
from ..clip.modeling_clip import CLIPOutput
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

"""PyTorch VisionTextDualEncoder model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def clip_loss(similarity: torch.Tensor) -> torch.Tensor: ...

class VisionTextDualEncoderModel(PreTrainedModel):
    config: VisionTextDualEncoderConfig
    base_model_prefix = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    def __init__(
        self,
        config: VisionTextDualEncoderConfig | None = ...,
        vision_model: PreTrainedModel | None = ...,
        text_model: PreTrainedModel | None = ...,
    ) -> None: ...
    def get_text_features(
        self,
        input_ids=...,
        attention_mask=...,
        position_ids=...,
        token_type_ids=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> Any:

        ...
    def get_image_features(
        self, pixel_values=..., output_attentions=..., output_hidden_states=..., return_dict=...
    ):  # -> Any:

        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | CLIPOutput: ...
    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str | None = ...,
        text_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel: ...

__all__ = ["VisionTextDualEncoderModel"]
