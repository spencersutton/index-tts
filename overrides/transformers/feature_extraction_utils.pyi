import os
from .dynamic_module_utils import custom_object_save as custom_object_save
from .utils import (
    FEATURE_EXTRACTOR_NAME as FEATURE_EXTRACTOR_NAME,
    PushToHubMixin as PushToHubMixin,
    TensorType as TensorType,
    add_model_info_to_auto_map as add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines as add_model_info_to_custom_pipelines,
    cached_file as cached_file,
    copy_func as copy_func,
    download_url as download_url,
    is_flax_available as is_flax_available,
    is_jax_tensor as is_jax_tensor,
    is_numpy_array as is_numpy_array,
    is_offline_mode as is_offline_mode,
    is_remote_url as is_remote_url,
    is_tf_available as is_tf_available,
    is_torch_available as is_torch_available,
    is_torch_device as is_torch_device,
    is_torch_dtype as is_torch_dtype,
    logging as logging,
    requires_backends as requires_backends,
)
from _typeshed import Incomplete
from collections import UserDict
from typing import Any, Self

logger: Incomplete
PreTrainedFeatureExtractor: Incomplete

class BatchFeature(UserDict):
    def __init__(self, data: dict[str, Any] | None = None, tensor_type: None | str | TensorType = None) -> None: ...
    def __getitem__(self, item: str) -> Any: ...
    def __getattr__(self, item: str): ...
    def keys(self): ...
    def values(self): ...
    def items(self): ...
    def convert_to_tensors(self, tensor_type: str | TensorType | None = None): ...
    data: Incomplete
    def to(self, *args, **kwargs) -> BatchFeature: ...

SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound=FeatureExtractionMixin)

class FeatureExtractionMixin(PushToHubMixin):
    def __init__(self, **kwargs) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> Self: ...
    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs): ...
    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...
    @classmethod
    def from_dict(cls, feature_extractor_dict: dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike) -> PreTrainedFeatureExtractor: ...
    def to_json_string(self) -> str: ...
    def to_json_file(self, json_file_path: str | os.PathLike): ...
    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoFeatureExtractor") -> None: ...
