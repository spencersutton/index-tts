from .utils import validate_hf_hub_args

logger = ...
README_TEMPLATE = ...
PYPROJECT_TEMPLATE = ...

@validate_hf_hub_args
def from_pretrained_fastai(repo_id: str, revision: str | None = ...): ...
@validate_hf_hub_args
def push_to_hub_fastai(
    learner,
    *,
    repo_id: str,
    commit_message: str = ...,
    private: bool | None = ...,
    token: str | None = ...,
    config: dict | None = ...,
    branch: str | None = ...,
    create_pr: bool | None = ...,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    delete_patterns: list[str] | str | None = ...,
    api_endpoint: str | None = ...,
):  # -> CommitInfo:

    ...
