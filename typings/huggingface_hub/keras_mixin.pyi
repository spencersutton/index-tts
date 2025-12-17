from pathlib import Path
from typing import Any

import tf_keras as keras
from huggingface_hub import ModelHubMixin
from huggingface_hub.utils import is_tf_available

from .utils import validate_hf_hub_args

logger = ...
keras = ...
if is_tf_available(): ...

@_requires_keras_2_model
def save_pretrained_keras(
    model,
    save_directory: str | Path,
    config: dict[str, Any] | None = ...,
    include_optimizer: bool = ...,
    plot_model: bool = ...,
    tags: list | str | None = ...,
    **model_save_kwargs,
):  # -> None:

    ...
def from_pretrained_keras(*args, **kwargs) -> KerasModelHubMixin: ...
@validate_hf_hub_args
@_requires_keras_2_model
def push_to_hub_keras(
    model,
    repo_id: str,
    *,
    config: dict | None = ...,
    commit_message: str = ...,
    private: bool | None = ...,
    api_endpoint: str | None = ...,
    token: str | None = ...,
    branch: str | None = ...,
    create_pr: bool | None = ...,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    delete_patterns: list[str] | str | None = ...,
    log_dir: str | None = ...,
    include_optimizer: bool = ...,
    tags: list | str | None = ...,
    plot_model: bool = ...,
    **model_save_kwargs,
):  # -> CommitInfo:

    ...

class KerasModelHubMixin(ModelHubMixin): ...
