from ...configuration_utils import PretrainedConfig

"""VitMatte model configuration"""
logger = ...

class VitMatteConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        backbone_config: PretrainedConfig = ...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        hidden_size: int = ...,
        batch_norm_eps: float = ...,
        initializer_range: float = ...,
        convstream_hidden_sizes: list[int] = ...,
        fusion_hidden_sizes: list[int] = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[str, type[PretrainedConfig]]:
        ...
    def to_dict(self):  # -> dict[str, Any]:

        ...

__all__ = ["VitMatteConfig"]
