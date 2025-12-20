from typing import Any

from .utils import validate_hf_hub_args
from .utils._deprecation import _deprecate_method

logger = ...
ALL_TASKS = ...

class InferenceApi:
    @validate_hf_hub_args
    @_deprecate_method(
        version="1.0",
        message=...,
    )
    def __init__(self, repo_id: str, task: str | None = ..., token: str | None = ..., gpu: bool = ...) -> None: ...
    def __call__(
        self,
        inputs: str | dict | list[str] | list[list[str]] | None = ...,
        params: dict | None = ...,
        data: bytes | None = ...,
        raw_response: bool = ...,
    ) -> Any: ...
