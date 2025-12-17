from typing import Any

from ...configuration_utils import PretrainedConfig
from ...utils import is_timm_available

"""Configuration for TimmWrapper models"""
if is_timm_available(): ...
logger = ...

class TimmWrapperConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        initializer_range: float = ...,
        do_pooling: bool = ...,
        model_args: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):  # -> Self:
        ...
    def to_dict(self) -> dict[str, Any]: ...

__all__ = ["TimmWrapperConfig"]
