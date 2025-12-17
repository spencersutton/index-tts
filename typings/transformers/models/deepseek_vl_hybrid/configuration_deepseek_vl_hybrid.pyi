from ...configuration_utils import PretrainedConfig
from ..auto import AutoConfig

logger = ...

class DeepseekVLHybridConfig(PretrainedConfig):
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

__all__ = ["DeepseekVLHybridConfig"]
