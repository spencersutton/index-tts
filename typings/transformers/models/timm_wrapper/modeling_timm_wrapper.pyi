from dataclasses import dataclass

import torch
from torch import Tensor

from ...modeling_outputs import ImageClassifierOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available
from .configuration_timm_wrapper import TimmWrapperConfig

if is_timm_available(): ...

@dataclass
class TimmWrapperModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

class TimmWrapperPreTrainedModel(PreTrainedModel):
    main_input_name = ...
    config: TimmWrapperConfig
    _no_split_modules = ...
    model_tags = ...
    accepts_loss_kwargs = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def post_init(self):  # -> None:
        ...
    def load_state_dict(self, state_dict, *args, **kwargs):  # -> _IncompatibleKeys:

        ...

class TimmWrapperModel(TimmWrapperPreTrainedModel):
    def __init__(self, config: TimmWrapperConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | list[int] | None = ...,
        return_dict: bool | None = ...,
        do_pooling: bool | None = ...,
        **kwargs,
    ) -> TimmWrapperModelOutput | tuple[Tensor, ...]: ...

class TimmWrapperForImageClassification(TimmWrapperPreTrainedModel):
    def __init__(self, config: TimmWrapperConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | list[int] | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> ImageClassifierOutput | tuple[Tensor, ...]: ...

__all__ = ["TimmWrapperForImageClassification", "TimmWrapperModel", "TimmWrapperPreTrainedModel"]
