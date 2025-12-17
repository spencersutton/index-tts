from dataclasses import dataclass
from typing import Any

logger = ...

@dataclass
class EvalResult:
    task_type: str
    dataset_type: str
    dataset_name: str
    metric_type: str
    metric_value: Any
    task_name: str | None = ...
    dataset_config: str | None = ...
    dataset_split: str | None = ...
    dataset_revision: str | None = ...
    dataset_args: dict[str, Any] | None = ...
    metric_name: str | None = ...
    metric_config: str | None = ...
    metric_args: dict[str, Any] | None = ...
    verified: bool | None = ...
    verify_token: str | None = ...
    source_name: str | None = ...
    source_url: str | None = ...
    @property
    def unique_identifier(self) -> tuple: ...
    def is_equal_except_value(self, other: EvalResult) -> bool: ...
    def __post_init__(self) -> None: ...

@dataclass
class CardData:
    def __init__(self, ignore_metadata_errors: bool = ..., **kwargs) -> None: ...
    def to_dict(self):  # -> dict[str, Any]:

        ...
    def to_yaml(self, line_break=..., original_order: list[str] | None = ...) -> str: ...
    def get(self, key: str, default: Any = ...) -> Any: ...
    def pop(self, key: str, default: Any = ...) -> Any: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...

class ModelCardData(CardData):
    def __init__(
        self,
        *,
        base_model: str | list[str] | None = ...,
        datasets: str | list[str] | None = ...,
        eval_results: list[EvalResult] | None = ...,
        language: str | list[str] | None = ...,
        library_name: str | None = ...,
        license: str | None = ...,
        license_name: str | None = ...,
        license_link: str | None = ...,
        metrics: list[str] | None = ...,
        model_name: str | None = ...,
        pipeline_tag: str | None = ...,
        tags: list[str] | None = ...,
        ignore_metadata_errors: bool = ...,
        **kwargs,
    ) -> None: ...

class DatasetCardData(CardData):
    def __init__(
        self,
        *,
        language: str | list[str] | None = ...,
        license: str | list[str] | None = ...,
        annotations_creators: str | list[str] | None = ...,
        language_creators: str | list[str] | None = ...,
        multilinguality: str | list[str] | None = ...,
        size_categories: str | list[str] | None = ...,
        source_datasets: list[str] | None = ...,
        task_categories: str | list[str] | None = ...,
        task_ids: str | list[str] | None = ...,
        paperswithcode_id: str | None = ...,
        pretty_name: str | None = ...,
        train_eval_index: dict | None = ...,
        config_names: str | list[str] | None = ...,
        ignore_metadata_errors: bool = ...,
        **kwargs,
    ) -> None: ...

class SpaceCardData(CardData):
    def __init__(
        self,
        *,
        title: str | None = ...,
        sdk: str | None = ...,
        sdk_version: str | None = ...,
        python_version: str | None = ...,
        app_file: str | None = ...,
        app_port: int | None = ...,
        license: str | None = ...,
        duplicated_from: str | None = ...,
        models: list[str] | None = ...,
        datasets: list[str] | None = ...,
        tags: list[str] | None = ...,
        ignore_metadata_errors: bool = ...,
        **kwargs,
    ) -> None: ...

def model_index_to_eval_results(model_index: list[dict[str, Any]]) -> tuple[str, list[EvalResult]]: ...
def eval_results_to_model_index(model_name: str, eval_results: list[EvalResult]) -> list[dict[str, Any]]: ...
