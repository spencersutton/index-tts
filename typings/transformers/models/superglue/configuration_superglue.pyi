from typing import TYPE_CHECKING

from ...configuration_utils import PretrainedConfig
from ..superpoint import SuperPointConfig

if TYPE_CHECKING: ...
logger = ...

class SuperGlueConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        keypoint_detector_config: SuperPointConfig = ...,
        hidden_size: int = ...,
        keypoint_encoder_sizes: list[int] | None = ...,
        gnn_layers_types: list[str] | None = ...,
        num_attention_heads: int = ...,
        sinkhorn_iterations: int = ...,
        matching_threshold: float = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[str, type[SuperPointConfig]]:
        ...

__all__ = ["SuperGlueConfig"]
