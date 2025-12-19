import inspect
from collections.abc import Callable
from dataclasses import Field, dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, TypeVar

from .repocard import ModelCard, ModelCardData
from .utils import is_safetensors_available, is_torch_available, validate_hf_hub_args

if is_torch_available(): ...
if is_safetensors_available(): ...
logger = ...

class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field]]

T = TypeVar("T", bound=ModelHubMixin)
ARGS_T = TypeVar("ARGS_T")
type ENCODER_T[ARGS_T] = Callable[[ARGS_T], Any]
type DECODER_T[ARGS_T] = Callable[[Any], ARGS_T]
type CODER_T = tuple[ENCODER_T, DECODER_T]
DEFAULT_MODEL_CARD = ...

@dataclass
class MixinInfo:
    model_card_template: str
    model_card_data: ModelCardData
    docs_url: str | None = ...
    paper_url: str | None = ...
    repo_url: str | None = ...

class ModelHubMixin:
    _hub_mixin_config: dict | DataclassInstance | None = ...
    _hub_mixin_info: MixinInfo
    _hub_mixin_inject_config: bool
    _hub_mixin_init_parameters: dict[str, inspect.Parameter]
    _hub_mixin_jsonable_default_values: dict[str, Any]
    _hub_mixin_jsonable_custom_types: tuple[type, ...]
    _hub_mixin_coders: dict[type, CODER_T]
    def __init_subclass__(
        cls,
        *,
        repo_url: str | None = ...,
        paper_url: str | None = ...,
        docs_url: str | None = ...,
        model_card_template: str = ...,
        language: list[str] | None = ...,
        library_name: str | None = ...,
        license: str | None = ...,
        license_name: str | None = ...,
        license_link: str | None = ...,
        pipeline_tag: str | None = ...,
        tags: list[str] | None = ...,
        coders: dict[type, CODER_T] | None = ...,
    ) -> None: ...
    def __new__(cls, *args, **kwargs) -> Self: ...
    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config: dict | DataclassInstance | None = ...,
        repo_id: str | None = ...,
        push_to_hub: bool = ...,
        model_card_kwargs: dict[str, Any] | None = ...,
        **push_to_hub_kwargs,
    ) -> str | None: ...
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *args: Any,
        force_download: bool = ...,
        resume_download: bool | None = ...,
        proxies: dict[object, object] | None = ...,
        token: str | bool | None = ...,
        cache_dir: str | Path | None = ...,
        local_files_only: bool = ...,
        revision: str | None = ...,
        **model_kwargs: Any,
    ) -> Self: ...
    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict | DataclassInstance | None = ...,
        commit_message: str = ...,
        private: bool | None = ...,
        token: str | None = ...,
        branch: str | None = ...,
        create_pr: bool | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        delete_patterns: list[str] | str | None = ...,
        model_card_kwargs: dict[str, Any] | None = ...,
    ) -> str: ...
    def generate_model_card(self, *args, **kwargs) -> ModelCard: ...

class PyTorchModelHubMixin(ModelHubMixin):
    def __init_subclass__(cls, *args, tags: list[str] | None = ..., **kwargs) -> None: ...
