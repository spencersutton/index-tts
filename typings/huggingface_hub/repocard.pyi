from pathlib import Path
from typing import Any

from huggingface_hub.repocard_data import CardData, DatasetCardData, ModelCardData, SpaceCardData

from .utils import validate_hf_hub_args

logger = ...
TEMPLATE_MODELCARD_PATH = ...
TEMPLATE_DATASETCARD_PATH = ...
REGEX_YAML_BLOCK = ...

class RepoCard:
    card_data_class = CardData
    default_template_path = ...
    repo_type = ...
    def __init__(self, content: str, ignore_metadata_errors: bool = ...) -> None: ...
    @property
    def content(self):  # -> str:

        ...
    @content.setter
    def content(self, content: str):  # -> None:

        ...
    def save(self, filepath: Path | str):  # -> None:

        ...
    @classmethod
    def load(
        cls,
        repo_id_or_path: str | Path,
        repo_type: str | None = ...,
        token: str | None = ...,
        ignore_metadata_errors: bool = ...,
    ):  # -> Self:

        ...
    def validate(self, repo_type: str | None = ...):  # -> None:

        ...
    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = ...,
        repo_type: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
    ):  # -> CommitInfo:

        ...
    @classmethod
    def from_template(
        cls,
        card_data: CardData,
        template_path: str | None = ...,
        template_str: str | None = ...,
        **template_kwargs,
    ):  # -> Self:

        ...

class ModelCard(RepoCard):
    card_data_class = ModelCardData
    default_template_path = ...
    repo_type = ...
    @classmethod
    def from_template(
        cls,
        card_data: ModelCardData,
        template_path: str | None = ...,
        template_str: str | None = ...,
        **template_kwargs,
    ):  # -> Self:

        ...

class DatasetCard(RepoCard):
    card_data_class = DatasetCardData
    default_template_path = ...
    repo_type = ...
    @classmethod
    def from_template(
        cls,
        card_data: DatasetCardData,
        template_path: str | None = ...,
        template_str: str | None = ...,
        **template_kwargs,
    ):  # -> Self:

        ...

class SpaceCard(RepoCard):
    card_data_class = SpaceCardData
    default_template_path = ...
    repo_type = ...

def metadata_load(local_path: str | Path) -> dict | None: ...
def metadata_save(local_path: str | Path, data: dict) -> None: ...
def metadata_eval_result(
    *,
    model_pretty_name: str,
    task_pretty_name: str,
    task_id: str,
    metrics_pretty_name: str,
    metrics_id: str,
    metrics_value: Any,
    dataset_pretty_name: str,
    dataset_id: str,
    metrics_config: str | None = ...,
    metrics_verified: bool = ...,
    dataset_config: str | None = ...,
    dataset_split: str | None = ...,
    dataset_revision: str | None = ...,
    metrics_verification_token: str | None = ...,
) -> dict: ...
@validate_hf_hub_args
def metadata_update(
    repo_id: str,
    metadata: dict,
    *,
    repo_type: str | None = ...,
    overwrite: bool = ...,
    token: str | None = ...,
    commit_message: str | None = ...,
    commit_description: str | None = ...,
    revision: str | None = ...,
    create_pr: bool = ...,
    parent_commit: str | None = ...,
) -> str: ...
