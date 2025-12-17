from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_vitpose import VitPoseConfig

"""PyTorch VitPose model."""
logger = ...

@dataclass
class VitPoseEstimatorOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    heatmaps: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class VitPosePreTrainedModel(PreTrainedModel):
    config: VitPoseConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

def flip_back(output_flipped, flip_pairs, target_type=...): ...

class VitPoseSimpleDecoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state: torch.Tensor, flip_pairs: torch.Tensor | None = ...) -> torch.Tensor: ...

class VitPoseClassicDecoder(nn.Module):
    def __init__(self, config: VitPoseConfig) -> None: ...
    def forward(self, hidden_state: torch.Tensor, flip_pairs: torch.Tensor | None = ...):  # -> Any:
        ...

class VitPoseForPoseEstimation(VitPosePreTrainedModel):
    def __init__(self, config: VitPoseConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: torch.Tensor | None = ...,
        flip_pairs: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | VitPoseEstimatorOutput: ...

__all__ = ["VitPoseForPoseEstimation", "VitPosePreTrainedModel"]
