from collections import UserDict, UserString
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Literal,
    TypeVar,
    overload,
)

from tqdm.auto import tqdm as base_tqdm

from . import constants
from ._commit_api import CommitOperation, CommitOperationAdd
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._jobs_api import JobInfo, JobSpec, ScheduledJobInfo
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
    Discussion,
    DiscussionComment,
    DiscussionStatusChange,
    DiscussionTitleChange,
    DiscussionWithDetails,
)
from .file_download import HfFileMetadata
from .inference._providers import PROVIDER_T
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import SafetensorsFileMetadata, SafetensorsRepoMetadata, experimental, validate_hf_hub_args
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT

R = TypeVar("R")
type CollectionItemType_T = Literal["model", "dataset", "space", "paper", "collection"]
type ExpandModelProperty_T = Literal[
    "author",
    "baseModels",
    "cardData",
    "childrenModelCount",
    "config",
    "createdAt",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "gated",
    "gguf",
    "inference",
    "inferenceProviderMapping",
    "lastModified",
    "library_name",
    "likes",
    "mask_token",
    "model-index",
    "pipeline_tag",
    "private",
    "resourceGroup",
    "safetensors",
    "sha",
    "siblings",
    "spaces",
    "tags",
    "transformersInfo",
    "trendingScore",
    "usedStorage",
    "widgetData",
    "xetEnabled",
]
type ExpandDatasetProperty_T = Literal[
    "author",
    "cardData",
    "citation",
    "createdAt",
    "description",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "gated",
    "lastModified",
    "likes",
    "paperswithcode_id",
    "private",
    "resourceGroup",
    "sha",
    "siblings",
    "tags",
    "trendingScore",
    "usedStorage",
    "xetEnabled",
]
type ExpandSpaceProperty_T = Literal[
    "author",
    "cardData",
    "createdAt",
    "datasets",
    "disabled",
    "lastModified",
    "likes",
    "models",
    "private",
    "resourceGroup",
    "runtime",
    "sdk",
    "sha",
    "siblings",
    "subdomain",
    "tags",
    "trendingScore",
    "usedStorage",
    "xetEnabled",
]
USERNAME_PLACEHOLDER = ...
_REGEX_DISCUSSION_URL = ...
_CREATE_COMMIT_NO_REPO_ERROR_MESSAGE = ...
_AUTH_CHECK_NO_REPO_ERROR_MESSAGE = ...
logger = ...

def repo_type_and_id_from_hf_id(hf_id: str, hub_url: str | None = ...) -> tuple[str | None, str | None, str]: ...

@dataclass
class LastCommitInfo(UserDict):
    oid: str
    title: str
    date: datetime
    def __post_init__(self):  # -> None:
        ...

@dataclass
class BlobLfsInfo(UserDict):
    size: int
    sha256: str
    pointer_size: int
    def __post_init__(self):  # -> None:
        ...

@dataclass
class BlobSecurityInfo(UserDict):
    safe: bool
    status: str
    av_scan: dict | None
    pickle_import_scan: dict | None
    def __post_init__(self):  # -> None:
        ...

@dataclass
class TransformersInfo(UserDict):
    auto_model: str
    custom_class: str | None = ...
    pipeline_tag: str | None = ...
    processor: str | None = ...
    def __post_init__(self):  # -> None:
        ...

@dataclass
class SafeTensorsInfo(UserDict):
    parameters: dict[str, int]
    total: int
    def __post_init__(self):  # -> None:
        ...

@dataclass
class CommitInfo(UserString):
    commit_url: str
    commit_message: str
    commit_description: str
    oid: str
    pr_url: str | None = ...
    repo_url: RepoUrl = ...
    pr_revision: str | None = ...
    pr_num: str | None = ...
    _url: str = ...
    def __new__(cls, *args, commit_url: str, _url: str | None = ..., **kwargs):  # -> Self:
        ...
    def __post_init__(self):  # -> None:
        ...

@dataclass
class AccessRequest:
    username: str
    fullname: str
    email: str | None
    timestamp: datetime
    status: Literal["pending", "accepted", "rejected"]
    fields: dict[str, Any] | None = ...

@dataclass
class WebhookWatchedItem:
    type: Literal["dataset", "model", "org", "space", "user"]
    name: str

@dataclass
class WebhookInfo:
    id: str
    url: str | None
    job: JobSpec | None
    watched: list[WebhookWatchedItem]
    domains: list[constants.WEBHOOK_DOMAIN_T]
    secret: str | None
    disabled: bool

class RepoUrl(UserString):
    def __new__(cls, url: Any, endpoint: str | None = ...):  # -> Self:
        ...
    def __init__(self, url: Any, endpoint: str | None = ...) -> None: ...

@dataclass
class RepoSibling:
    rfilename: str
    size: int | None = ...
    blob_id: str | None = ...
    lfs: BlobLfsInfo | None = ...

@dataclass
class RepoFile:
    path: str
    size: int
    blob_id: str
    lfs: BlobLfsInfo | None = ...
    last_commit: LastCommitInfo | None = ...
    security: BlobSecurityInfo | None = ...
    def __init__(self, **kwargs) -> None: ...

@dataclass
class RepoFolder:
    path: str
    tree_id: str
    last_commit: LastCommitInfo | None = ...
    def __init__(self, **kwargs) -> None: ...

@dataclass
class InferenceProviderMapping:
    provider: PROVIDER_T
    hf_model_id: str
    provider_id: str
    status: Literal["error", "live", "staging"]
    task: str
    adapter: str | None = ...
    adapter_weights_path: str | None = ...
    type: Literal["single-model", "tag-filter"] | None = ...
    def __init__(self, **kwargs) -> None: ...

@dataclass
class ModelInfo:
    id: str
    author: str | None
    sha: str | None
    created_at: datetime | None
    last_modified: datetime | None
    private: bool | None
    disabled: bool | None
    downloads: int | None
    downloads_all_time: int | None
    gated: Literal["auto", "manual", False] | None
    gguf: dict | None
    inference: Literal["warm"] | None
    inference_provider_mapping: list[InferenceProviderMapping] | None
    likes: int | None
    library_name: str | None
    tags: list[str] | None
    pipeline_tag: str | None
    mask_token: str | None
    card_data: ModelCardData | None
    widget_data: Any | None
    model_index: dict | None
    config: dict | None
    transformers_info: TransformersInfo | None
    trending_score: int | None
    siblings: list[RepoSibling] | None
    spaces: list[str] | None
    safetensors: SafeTensorsInfo | None
    security_repo_status: dict | None
    xet_enabled: bool | None
    def __init__(self, **kwargs) -> None: ...

@dataclass
class DatasetInfo:
    id: str
    author: str | None
    sha: str | None
    created_at: datetime | None
    last_modified: datetime | None
    private: bool | None
    gated: Literal["auto", "manual", False] | None
    disabled: bool | None
    downloads: int | None
    downloads_all_time: int | None
    likes: int | None
    paperswithcode_id: str | None
    tags: list[str] | None
    trending_score: int | None
    card_data: DatasetCardData | None
    siblings: list[RepoSibling] | None
    xet_enabled: bool | None
    def __init__(self, **kwargs) -> None: ...

@dataclass
class SpaceInfo:
    id: str
    author: str | None
    sha: str | None
    created_at: datetime | None
    last_modified: datetime | None
    private: bool | None
    gated: Literal["auto", "manual", False] | None
    disabled: bool | None
    host: str | None
    subdomain: str | None
    likes: int | None
    sdk: str | None
    tags: list[str] | None
    siblings: list[RepoSibling] | None
    trending_score: int | None
    card_data: SpaceCardData | None
    runtime: SpaceRuntime | None
    models: list[str] | None
    datasets: list[str] | None
    xet_enabled: bool | None
    def __init__(self, **kwargs) -> None: ...

@dataclass
class CollectionItem:
    item_object_id: str
    item_id: str
    item_type: str
    position: int
    note: str | None = ...
    def __init__(
        self, _id: str, id: str, type: CollectionItemType_T, position: int, note: dict | None = ..., **kwargs
    ) -> None: ...

@dataclass
class Collection:
    slug: str
    title: str
    owner: str
    items: list[CollectionItem]
    last_updated: datetime
    position: int
    private: bool
    theme: str
    upvotes: int
    description: str | None = ...
    def __init__(self, **kwargs) -> None: ...
    @property
    def url(self) -> str: ...

@dataclass
class GitRefInfo:
    name: str
    ref: str
    target_commit: str

@dataclass
class GitRefs:
    branches: list[GitRefInfo]
    converts: list[GitRefInfo]
    tags: list[GitRefInfo]
    pull_requests: list[GitRefInfo] | None = ...

@dataclass
class GitCommitInfo:
    commit_id: str
    authors: list[str]
    created_at: datetime
    title: str
    message: str
    formatted_title: str | None
    formatted_message: str | None

@dataclass
class UserLikes:
    user: str
    total: int
    datasets: list[str]
    models: list[str]
    spaces: list[str]

@dataclass
class Organization:
    avatar_url: str
    name: str
    fullname: str
    details: str | None = ...
    is_verified: bool | None = ...
    is_following: bool | None = ...
    num_users: int | None = ...
    num_models: int | None = ...
    num_spaces: int | None = ...
    num_datasets: int | None = ...
    num_followers: int | None = ...
    def __init__(self, **kwargs) -> None: ...

@dataclass
class User:
    username: str
    fullname: str
    avatar_url: str
    details: str | None = ...
    is_following: bool | None = ...
    is_pro: bool | None = ...
    num_models: int | None = ...
    num_datasets: int | None = ...
    num_spaces: int | None = ...
    num_discussions: int | None = ...
    num_papers: int | None = ...
    num_upvotes: int | None = ...
    num_likes: int | None = ...
    num_following: int | None = ...
    num_followers: int | None = ...
    orgs: list[Organization] = ...
    def __init__(self, **kwargs) -> None: ...

@dataclass
class PaperInfo:
    id: str
    authors: list[str] | None
    published_at: datetime | None
    title: str | None
    summary: str | None
    upvotes: int | None
    discussion_id: str | None
    source: str | None
    comments: int | None
    submitted_at: datetime | None
    submitted_by: User | None
    def __init__(self, **kwargs) -> None: ...

@dataclass
class LFSFileInfo:
    file_oid: str
    filename: str
    oid: str
    pushed_at: datetime
    ref: str | None
    size: int
    def __init__(self, **kwargs) -> None: ...

def future_compatible(fn: CallableT) -> CallableT: ...

class HfApi:
    def __init__(
        self,
        endpoint: str | None = ...,
        token: str | bool | None = ...,
        library_name: str | None = ...,
        library_version: str | None = ...,
        user_agent: dict | str | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> None: ...
    def run_as_future(self, fn: Callable[..., R], *args, **kwargs) -> Future[R]: ...
    @validate_hf_hub_args
    def whoami(self, token: bool | str | None = ...) -> dict: ...
    @_deprecate_method(
        version="1.0",
        message=...,
    )
    def get_token_permission(
        self, token: bool | str | None = ...
    ) -> Literal["read", "write", "fineGrained"] | None: ...
    def get_model_tags(self) -> dict: ...
    def get_dataset_tags(self) -> dict: ...
    @_deprecate_arguments(
        version="1.0", deprecated_args=["language", "library", "task", "tags"], custom_message="Use `filter` instead."
    )
    @validate_hf_hub_args
    def list_models(
        self,
        *,
        filter: str | Iterable[str] | None = ...,
        author: str | None = ...,
        apps: str | list[str] | None = ...,
        gated: bool | None = ...,
        inference: Literal["warm"] | None = ...,
        inference_provider: Literal["all"] | PROVIDER_T | list[PROVIDER_T] | None = ...,
        model_name: str | None = ...,
        trained_dataset: str | list[str] | None = ...,
        search: str | None = ...,
        pipeline_tag: str | None = ...,
        emissions_thresholds: tuple[float, float] | None = ...,
        sort: Literal["last_modified"] | str | None = ...,
        direction: Literal[-1] | None = ...,
        limit: int | None = ...,
        expand: list[ExpandModelProperty_T] | None = ...,
        full: bool | None = ...,
        cardData: bool = ...,
        fetch_config: bool = ...,
        token: bool | str | None = ...,
        language: str | list[str] | None = ...,
        library: str | list[str] | None = ...,
        tags: str | list[str] | None = ...,
        task: str | list[str] | None = ...,
    ) -> Iterable[ModelInfo]: ...
    @_deprecate_arguments(version="1.0", deprecated_args=["tags"], custom_message="Use `filter` instead.")
    @validate_hf_hub_args
    def list_datasets(
        self,
        *,
        filter: str | Iterable[str] | None = ...,
        author: str | None = ...,
        benchmark: str | list[str] | None = ...,
        dataset_name: str | None = ...,
        gated: bool | None = ...,
        language_creators: str | list[str] | None = ...,
        language: str | list[str] | None = ...,
        multilinguality: str | list[str] | None = ...,
        size_categories: str | list[str] | None = ...,
        task_categories: str | list[str] | None = ...,
        task_ids: str | list[str] | None = ...,
        search: str | None = ...,
        sort: Literal["last_modified"] | str | None = ...,
        direction: Literal[-1] | None = ...,
        limit: int | None = ...,
        expand: list[ExpandDatasetProperty_T] | None = ...,
        full: bool | None = ...,
        token: bool | str | None = ...,
        tags: str | list[str] | None = ...,
    ) -> Iterable[DatasetInfo]: ...
    @validate_hf_hub_args
    def list_spaces(
        self,
        *,
        filter: str | Iterable[str] | None = ...,
        author: str | None = ...,
        search: str | None = ...,
        datasets: str | Iterable[str] | None = ...,
        models: str | Iterable[str] | None = ...,
        linked: bool = ...,
        sort: Literal["last_modified"] | str | None = ...,
        direction: Literal[-1] | None = ...,
        limit: int | None = ...,
        expand: list[ExpandSpaceProperty_T] | None = ...,
        full: bool | None = ...,
        token: bool | str | None = ...,
    ) -> Iterable[SpaceInfo]: ...
    @validate_hf_hub_args
    def unlike(self, repo_id: str, *, token: bool | str | None = ..., repo_type: str | None = ...) -> None: ...
    @validate_hf_hub_args
    def list_liked_repos(self, user: str | None = ..., *, token: bool | str | None = ...) -> UserLikes: ...
    @validate_hf_hub_args
    def list_repo_likers(
        self, repo_id: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> Iterable[User]: ...
    @validate_hf_hub_args
    def model_info(
        self,
        repo_id: str,
        *,
        revision: str | None = ...,
        timeout: float | None = ...,
        securityStatus: bool | None = ...,
        files_metadata: bool = ...,
        expand: list[ExpandModelProperty_T] | None = ...,
        token: bool | str | None = ...,
    ) -> ModelInfo: ...
    @validate_hf_hub_args
    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: str | None = ...,
        timeout: float | None = ...,
        files_metadata: bool = ...,
        expand: list[ExpandDatasetProperty_T] | None = ...,
        token: bool | str | None = ...,
    ) -> DatasetInfo: ...
    @validate_hf_hub_args
    def space_info(
        self,
        repo_id: str,
        *,
        revision: str | None = ...,
        timeout: float | None = ...,
        files_metadata: bool = ...,
        expand: list[ExpandSpaceProperty_T] | None = ...,
        token: bool | str | None = ...,
    ) -> SpaceInfo: ...
    @validate_hf_hub_args
    def repo_info(
        self,
        repo_id: str,
        *,
        revision: str | None = ...,
        repo_type: str | None = ...,
        timeout: float | None = ...,
        files_metadata: bool = ...,
        expand: ExpandModelProperty_T | ExpandDatasetProperty_T | ExpandSpaceProperty_T | None = ...,
        token: bool | str | None = ...,
    ) -> ModelInfo | DatasetInfo | SpaceInfo: ...
    @validate_hf_hub_args
    def repo_exists(self, repo_id: str, *, repo_type: str | None = ..., token: str | bool | None = ...) -> bool: ...
    @validate_hf_hub_args
    def revision_exists(
        self, repo_id: str, revision: str, *, repo_type: str | None = ..., token: str | bool | None = ...
    ) -> bool: ...
    @validate_hf_hub_args
    def file_exists(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = ...,
        revision: str | None = ...,
        token: str | bool | None = ...,
    ) -> bool: ...
    @validate_hf_hub_args
    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str | None = ...,
        repo_type: str | None = ...,
        token: str | bool | None = ...,
    ) -> list[str]: ...
    @validate_hf_hub_args
    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = ...,
        *,
        recursive: bool = ...,
        expand: bool = ...,
        revision: str | None = ...,
        repo_type: str | None = ...,
        token: str | bool | None = ...,
    ) -> Iterable[RepoFile | RepoFolder]: ...
    @validate_hf_hub_args
    def list_repo_refs(
        self,
        repo_id: str,
        *,
        repo_type: str | None = ...,
        include_pull_requests: bool = ...,
        token: str | bool | None = ...,
    ) -> GitRefs: ...
    @validate_hf_hub_args
    def list_repo_commits(
        self,
        repo_id: str,
        *,
        repo_type: str | None = ...,
        token: bool | str | None = ...,
        revision: str | None = ...,
        formatted: bool = ...,
    ) -> list[GitCommitInfo]: ...
    @validate_hf_hub_args
    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str] | str,
        *,
        expand: bool = ...,
        revision: str | None = ...,
        repo_type: str | None = ...,
        token: str | bool | None = ...,
    ) -> list[RepoFile | RepoFolder]: ...
    @validate_hf_hub_args
    def super_squash_history(
        self,
        repo_id: str,
        *,
        branch: str | None = ...,
        commit_message: str | None = ...,
        repo_type: str | None = ...,
        token: str | bool | None = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def list_lfs_files(
        self, repo_id: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> Iterable[LFSFileInfo]: ...
    @validate_hf_hub_args
    def permanently_delete_lfs_files(
        self,
        repo_id: str,
        lfs_files: Iterable[LFSFileInfo],
        *,
        rewrite_history: bool = ...,
        repo_type: str | None = ...,
        token: bool | str | None = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def create_repo(
        self,
        repo_id: str,
        *,
        token: str | bool | None = ...,
        private: bool | None = ...,
        repo_type: str | None = ...,
        exist_ok: bool = ...,
        resource_group_id: str | None = ...,
        space_sdk: str | None = ...,
        space_hardware: SpaceHardware | None = ...,
        space_storage: SpaceStorage | None = ...,
        space_sleep_time: int | None = ...,
        space_secrets: list[dict[str, str]] | None = ...,
        space_variables: list[dict[str, str]] | None = ...,
    ) -> RepoUrl: ...
    @validate_hf_hub_args
    def delete_repo(
        self,
        repo_id: str,
        *,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        missing_ok: bool = ...,
    ) -> None: ...
    @_deprecate_method(version="0.32", message="Please use `update_repo_settings` instead.")
    @validate_hf_hub_args
    def update_repo_visibility(
        self, repo_id: str, private: bool = ..., *, token: str | bool | None = ..., repo_type: str | None = ...
    ) -> dict[str, bool]: ...
    @validate_hf_hub_args
    def update_repo_settings(
        self,
        repo_id: str,
        *,
        gated: Literal["auto", "manual", False] | None = ...,
        private: bool | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        xet_enabled: bool | None = ...,
    ) -> None: ...
    def move_repo(
        self, from_id: str, to_id: str, *, repo_type: str | None = ..., token: str | bool | None = ...
    ):  # -> None:
        ...
    @overload
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        num_threads: int = ...,
        parent_commit: str | None = ...,
        run_as_future: Literal[False] = ...,
    ) -> CommitInfo: ...
    @overload
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        num_threads: int = ...,
        parent_commit: str | None = ...,
        run_as_future: Literal[True] = ...,
    ) -> Future[CommitInfo]: ...
    @validate_hf_hub_args
    @future_compatible
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        num_threads: int = ...,
        parent_commit: str | None = ...,
        run_as_future: bool = ...,
    ) -> CommitInfo | Future[CommitInfo]: ...
    def preupload_lfs_files(
        self,
        repo_id: str,
        additions: Iterable[CommitOperationAdd],
        *,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        num_threads: int = ...,
        free_memory: bool = ...,
        gitignore_content: str | None = ...,
    ):  # -> None:
        ...
    @overload
    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes | BinaryIO,
        path_in_repo: str,
        repo_id: str,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
        run_as_future: Literal[False] = ...,
    ) -> CommitInfo: ...
    @overload
    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes | BinaryIO,
        path_in_repo: str,
        repo_id: str,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
        run_as_future: Literal[True] = ...,
    ) -> Future[CommitInfo]: ...
    @validate_hf_hub_args
    @future_compatible
    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes | BinaryIO,
        path_in_repo: str,
        repo_id: str,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
        run_as_future: bool = ...,
    ) -> CommitInfo | Future[CommitInfo]: ...
    @overload
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        delete_patterns: list[str] | str | None = ...,
        run_as_future: Literal[False] = ...,
    ) -> CommitInfo: ...
    @overload
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        delete_patterns: list[str] | str | None = ...,
        run_as_future: Literal[True] = ...,
    ) -> Future[CommitInfo]: ...
    @validate_hf_hub_args
    @future_compatible
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        delete_patterns: list[str] | str | None = ...,
        run_as_future: bool = ...,
    ) -> CommitInfo | Future[CommitInfo]: ...
    @validate_hf_hub_args
    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: str | bool | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
    ) -> CommitInfo: ...
    @validate_hf_hub_args
    def delete_files(
        self,
        repo_id: str,
        delete_patterns: list[str],
        *,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
    ) -> CommitInfo: ...
    @validate_hf_hub_args
    def delete_folder(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        commit_message: str | None = ...,
        commit_description: str | None = ...,
        create_pr: bool | None = ...,
        parent_commit: str | None = ...,
    ) -> CommitInfo: ...
    def upload_large_folder(
        self,
        repo_id: str,
        folder_path: str | Path,
        *,
        repo_type: str,
        revision: str | None = ...,
        private: bool | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        num_workers: int | None = ...,
        print_report: bool = ...,
        print_report_every: int = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def get_hf_file_metadata(
        self,
        *,
        url: str,
        token: bool | str | None = ...,
        proxies: dict | None = ...,
        timeout: float | None = ...,
    ) -> HfFileMetadata: ...
    @validate_hf_hub_args
    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        subfolder: str | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        cache_dir: str | Path | None = ...,
        local_dir: str | Path | None = ...,
        force_download: bool = ...,
        proxies: dict | None = ...,
        etag_timeout: float = ...,
        token: bool | str | None = ...,
        local_files_only: bool = ...,
        resume_download: bool | None = ...,
        force_filename: str | None = ...,
        local_dir_use_symlinks: bool | Literal["auto"] = ...,
    ) -> str: ...
    @validate_hf_hub_args
    def snapshot_download(
        self,
        repo_id: str,
        *,
        repo_type: str | None = ...,
        revision: str | None = ...,
        cache_dir: str | Path | None = ...,
        local_dir: str | Path | None = ...,
        proxies: dict | None = ...,
        etag_timeout: float = ...,
        force_download: bool = ...,
        token: bool | str | None = ...,
        local_files_only: bool = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        max_workers: int = ...,
        tqdm_class: type[base_tqdm] | None = ...,
        local_dir_use_symlinks: bool | Literal["auto"] = ...,
        resume_download: bool | None = ...,
    ) -> str: ...
    def get_safetensors_metadata(
        self,
        repo_id: str,
        *,
        repo_type: str | None = ...,
        revision: str | None = ...,
        token: bool | str | None = ...,
    ) -> SafetensorsRepoMetadata: ...
    def parse_safetensors_file_metadata(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = ...,
        revision: str | None = ...,
        token: bool | str | None = ...,
    ) -> SafetensorsFileMetadata: ...
    @validate_hf_hub_args
    def create_branch(
        self,
        repo_id: str,
        *,
        branch: str,
        revision: str | None = ...,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
        exist_ok: bool = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def delete_branch(
        self, repo_id: str, *, branch: str, token: bool | str | None = ..., repo_type: str | None = ...
    ) -> None: ...
    @validate_hf_hub_args
    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        tag_message: str | None = ...,
        revision: str | None = ...,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
        exist_ok: bool = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def delete_tag(
        self, repo_id: str, *, tag: str, token: bool | str | None = ..., repo_type: str | None = ...
    ) -> None: ...
    @validate_hf_hub_args
    def get_full_repo_name(
        self, model_id: str, *, organization: str | None = ..., token: bool | str | None = ...
    ):  # -> str:
        ...
    @validate_hf_hub_args
    def get_repo_discussions(
        self,
        repo_id: str,
        *,
        author: str | None = ...,
        discussion_type: constants.DiscussionTypeFilter | None = ...,
        discussion_status: constants.DiscussionStatusFilter | None = ...,
        repo_type: str | None = ...,
        token: bool | str | None = ...,
    ) -> Iterator[Discussion]: ...
    @validate_hf_hub_args
    def get_discussion_details(
        self, repo_id: str, discussion_num: int, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> DiscussionWithDetails: ...
    @validate_hf_hub_args
    def create_discussion(
        self,
        repo_id: str,
        title: str,
        *,
        token: bool | str | None = ...,
        description: str | None = ...,
        repo_type: str | None = ...,
        pull_request: bool = ...,
    ) -> DiscussionWithDetails: ...
    @validate_hf_hub_args
    def create_pull_request(
        self,
        repo_id: str,
        title: str,
        *,
        token: bool | str | None = ...,
        description: str | None = ...,
        repo_type: str | None = ...,
    ) -> DiscussionWithDetails: ...
    @validate_hf_hub_args
    def comment_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        comment: str,
        *,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
    ) -> DiscussionComment: ...
    @validate_hf_hub_args
    def rename_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        new_title: str,
        *,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
    ) -> DiscussionTitleChange: ...
    @validate_hf_hub_args
    def change_discussion_status(
        self,
        repo_id: str,
        discussion_num: int,
        new_status: Literal["open", "closed"],
        *,
        token: bool | str | None = ...,
        comment: str | None = ...,
        repo_type: str | None = ...,
    ) -> DiscussionStatusChange: ...
    @validate_hf_hub_args
    def merge_pull_request(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        token: bool | str | None = ...,
        comment: str | None = ...,
        repo_type: str | None = ...,
    ):  # -> None:
        ...
    @validate_hf_hub_args
    def edit_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        new_content: str,
        *,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
    ) -> DiscussionComment: ...
    @validate_hf_hub_args
    def hide_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        *,
        token: bool | str | None = ...,
        repo_type: str | None = ...,
    ) -> DiscussionComment: ...
    @validate_hf_hub_args
    def add_space_secret(
        self,
        repo_id: str,
        key: str,
        value: str,
        *,
        description: str | None = ...,
        token: bool | str | None = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def delete_space_secret(self, repo_id: str, key: str, *, token: bool | str | None = ...) -> None: ...
    @validate_hf_hub_args
    def get_space_variables(self, repo_id: str, *, token: bool | str | None = ...) -> dict[str, SpaceVariable]: ...
    @validate_hf_hub_args
    def add_space_variable(
        self,
        repo_id: str,
        key: str,
        value: str,
        *,
        description: str | None = ...,
        token: bool | str | None = ...,
    ) -> dict[str, SpaceVariable]: ...
    @validate_hf_hub_args
    def delete_space_variable(
        self, repo_id: str, key: str, *, token: bool | str | None = ...
    ) -> dict[str, SpaceVariable]: ...
    @validate_hf_hub_args
    def get_space_runtime(self, repo_id: str, *, token: bool | str | None = ...) -> SpaceRuntime: ...
    @validate_hf_hub_args
    def request_space_hardware(
        self,
        repo_id: str,
        hardware: SpaceHardware,
        *,
        token: bool | str | None = ...,
        sleep_time: int | None = ...,
    ) -> SpaceRuntime: ...
    @validate_hf_hub_args
    def set_space_sleep_time(
        self, repo_id: str, sleep_time: int, *, token: bool | str | None = ...
    ) -> SpaceRuntime: ...
    @validate_hf_hub_args
    def pause_space(self, repo_id: str, *, token: bool | str | None = ...) -> SpaceRuntime: ...
    @validate_hf_hub_args
    def restart_space(
        self, repo_id: str, *, token: bool | str | None = ..., factory_reboot: bool = ...
    ) -> SpaceRuntime: ...
    @validate_hf_hub_args
    def duplicate_space(
        self,
        from_id: str,
        to_id: str | None = ...,
        *,
        private: bool | None = ...,
        token: bool | str | None = ...,
        exist_ok: bool = ...,
        hardware: SpaceHardware | None = ...,
        storage: SpaceStorage | None = ...,
        sleep_time: int | None = ...,
        secrets: list[dict[str, str]] | None = ...,
        variables: list[dict[str, str]] | None = ...,
    ) -> RepoUrl: ...
    @validate_hf_hub_args
    def request_space_storage(
        self, repo_id: str, storage: SpaceStorage, *, token: bool | str | None = ...
    ) -> SpaceRuntime: ...
    @validate_hf_hub_args
    def delete_space_storage(self, repo_id: str, *, token: bool | str | None = ...) -> SpaceRuntime: ...
    def list_inference_endpoints(
        self, namespace: str | None = ..., *, token: bool | str | None = ...
    ) -> list[InferenceEndpoint]: ...
    def create_inference_endpoint(
        self,
        name: str,
        *,
        repository: str,
        framework: str,
        accelerator: str,
        instance_size: str,
        instance_type: str,
        region: str,
        vendor: str,
        account_id: str | None = ...,
        min_replica: int = ...,
        max_replica: int = ...,
        scale_to_zero_timeout: int | None = ...,
        revision: str | None = ...,
        task: str | None = ...,
        custom_image: dict | None = ...,
        env: dict[str, str] | None = ...,
        secrets: dict[str, str] | None = ...,
        type: InferenceEndpointType = ...,
        domain: str | None = ...,
        path: str | None = ...,
        cache_http_responses: bool | None = ...,
        tags: list[str] | None = ...,
        namespace: str | None = ...,
        token: bool | str | None = ...,
    ) -> InferenceEndpoint: ...
    @experimental
    @validate_hf_hub_args
    def create_inference_endpoint_from_catalog(
        self,
        repo_id: str,
        *,
        name: str | None = ...,
        token: bool | str | None = ...,
        namespace: str | None = ...,
    ) -> InferenceEndpoint: ...
    @experimental
    @validate_hf_hub_args
    def list_inference_catalog(self, *, token: bool | str | None = ...) -> list[str]: ...
    def get_inference_endpoint(
        self, name: str, *, namespace: str | None = ..., token: bool | str | None = ...
    ) -> InferenceEndpoint: ...
    def update_inference_endpoint(
        self,
        name: str,
        *,
        accelerator: str | None = ...,
        instance_size: str | None = ...,
        instance_type: str | None = ...,
        min_replica: int | None = ...,
        max_replica: int | None = ...,
        scale_to_zero_timeout: int | None = ...,
        repository: str | None = ...,
        framework: str | None = ...,
        revision: str | None = ...,
        task: str | None = ...,
        custom_image: dict | None = ...,
        env: dict[str, str] | None = ...,
        secrets: dict[str, str] | None = ...,
        domain: str | None = ...,
        path: str | None = ...,
        cache_http_responses: bool | None = ...,
        tags: list[str] | None = ...,
        namespace: str | None = ...,
        token: bool | str | None = ...,
    ) -> InferenceEndpoint: ...
    def delete_inference_endpoint(
        self, name: str, *, namespace: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    def pause_inference_endpoint(
        self, name: str, *, namespace: str | None = ..., token: bool | str | None = ...
    ) -> InferenceEndpoint: ...
    def resume_inference_endpoint(
        self, name: str, *, namespace: str | None = ..., running_ok: bool = ..., token: bool | str | None = ...
    ) -> InferenceEndpoint: ...
    def scale_to_zero_inference_endpoint(
        self, name: str, *, namespace: str | None = ..., token: bool | str | None = ...
    ) -> InferenceEndpoint: ...
    @validate_hf_hub_args
    def list_collections(
        self,
        *,
        owner: list[str] | str | None = ...,
        item: list[str] | str | None = ...,
        sort: Literal["lastModified", "trending", "upvotes"] | None = ...,
        limit: int | None = ...,
        token: bool | str | None = ...,
    ) -> Iterable[Collection]: ...
    def get_collection(self, collection_slug: str, *, token: bool | str | None = ...) -> Collection: ...
    def create_collection(
        self,
        title: str,
        *,
        namespace: str | None = ...,
        description: str | None = ...,
        private: bool = ...,
        exists_ok: bool = ...,
        token: bool | str | None = ...,
    ) -> Collection: ...
    def update_collection_metadata(
        self,
        collection_slug: str,
        *,
        title: str | None = ...,
        description: str | None = ...,
        position: int | None = ...,
        private: bool | None = ...,
        theme: str | None = ...,
        token: bool | str | None = ...,
    ) -> Collection: ...
    def delete_collection(
        self, collection_slug: str, *, missing_ok: bool = ..., token: bool | str | None = ...
    ) -> None: ...
    def add_collection_item(
        self,
        collection_slug: str,
        item_id: str,
        item_type: CollectionItemType_T,
        *,
        note: str | None = ...,
        exists_ok: bool = ...,
        token: bool | str | None = ...,
    ) -> Collection: ...
    def update_collection_item(
        self,
        collection_slug: str,
        item_object_id: str,
        *,
        note: str | None = ...,
        position: int | None = ...,
        token: bool | str | None = ...,
    ) -> None: ...
    def delete_collection_item(
        self, collection_slug: str, item_object_id: str, *, missing_ok: bool = ..., token: bool | str | None = ...
    ) -> None: ...
    @validate_hf_hub_args
    def list_pending_access_requests(
        self, repo_id: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> list[AccessRequest]: ...
    @validate_hf_hub_args
    def list_accepted_access_requests(
        self, repo_id: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> list[AccessRequest]: ...
    @validate_hf_hub_args
    def list_rejected_access_requests(
        self, repo_id: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> list[AccessRequest]: ...
    @validate_hf_hub_args
    def cancel_access_request(
        self, repo_id: str, user: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    @validate_hf_hub_args
    def accept_access_request(
        self, repo_id: str, user: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    @validate_hf_hub_args
    def reject_access_request(
        self,
        repo_id: str,
        user: str,
        *,
        repo_type: str | None = ...,
        rejection_reason: str | None,
        token: bool | str | None = ...,
    ) -> None: ...
    @validate_hf_hub_args
    def grant_access(
        self, repo_id: str, user: str, *, repo_type: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    @validate_hf_hub_args
    def get_webhook(self, webhook_id: str, *, token: bool | str | None = ...) -> WebhookInfo: ...
    @validate_hf_hub_args
    def list_webhooks(self, *, token: bool | str | None = ...) -> list[WebhookInfo]: ...
    @validate_hf_hub_args
    def create_webhook(
        self,
        *,
        url: str | None = ...,
        job_id: str | None = ...,
        watched: list[dict | WebhookWatchedItem],
        domains: list[constants.WEBHOOK_DOMAIN_T] | None = ...,
        secret: str | None = ...,
        token: bool | str | None = ...,
    ) -> WebhookInfo: ...
    @validate_hf_hub_args
    def update_webhook(
        self,
        webhook_id: str,
        *,
        url: str | None = ...,
        watched: list[dict | WebhookWatchedItem] | None = ...,
        domains: list[constants.WEBHOOK_DOMAIN_T] | None = ...,
        secret: str | None = ...,
        token: bool | str | None = ...,
    ) -> WebhookInfo: ...
    @validate_hf_hub_args
    def enable_webhook(self, webhook_id: str, *, token: bool | str | None = ...) -> WebhookInfo: ...
    @validate_hf_hub_args
    def disable_webhook(self, webhook_id: str, *, token: bool | str | None = ...) -> WebhookInfo: ...
    @validate_hf_hub_args
    def delete_webhook(self, webhook_id: str, *, token: bool | str | None = ...) -> None: ...
    def get_user_overview(self, username: str, token: bool | str | None = ...) -> User: ...
    @validate_hf_hub_args
    def get_organization_overview(self, organization: str, token: bool | str | None = ...) -> Organization: ...
    def list_organization_members(self, organization: str, token: bool | str | None = ...) -> Iterable[User]: ...
    def list_user_followers(self, username: str, token: bool | str | None = ...) -> Iterable[User]: ...
    def list_user_following(self, username: str, token: bool | str | None = ...) -> Iterable[User]: ...
    def list_papers(self, *, query: str | None = ..., token: bool | str | None = ...) -> Iterable[PaperInfo]: ...
    def paper_info(self, id: str) -> PaperInfo: ...
    def auth_check(self, repo_id: str, *, repo_type: str | None = ..., token: bool | str | None = ...) -> None: ...
    def run_job(
        self,
        *,
        image: str,
        command: list[str],
        env: dict[str, Any] | None = ...,
        secrets: dict[str, Any] | None = ...,
        flavor: SpaceHardware | None = ...,
        timeout: float | str | None = ...,
        namespace: str | None = ...,
        token: bool | str | None = ...,
    ) -> JobInfo: ...
    def fetch_job_logs(
        self, *, job_id: str, namespace: str | None = ..., token: bool | str | None = ...
    ) -> Iterable[str]: ...
    def list_jobs(
        self, *, timeout: int | None = ..., namespace: str | None = ..., token: bool | str | None = ...
    ) -> list[JobInfo]: ...
    def inspect_job(self, *, job_id: str, namespace: str | None = ..., token: bool | str | None = ...) -> JobInfo: ...
    def cancel_job(self, *, job_id: str, namespace: str | None = ..., token: bool | str | None = ...) -> None: ...
    @experimental
    def run_uv_job(
        self,
        script: str,
        *,
        script_args: list[str] | None = ...,
        dependencies: list[str] | None = ...,
        python: str | None = ...,
        image: str | None = ...,
        env: dict[str, Any] | None = ...,
        secrets: dict[str, Any] | None = ...,
        flavor: SpaceHardware | None = ...,
        timeout: float | str | None = ...,
        namespace: str | None = ...,
        token: bool | str | None = ...,
        _repo: str | None = ...,
    ) -> JobInfo: ...
    def create_scheduled_job(
        self,
        *,
        image: str,
        command: list[str],
        schedule: str,
        suspend: bool | None = ...,
        concurrency: bool | None = ...,
        env: dict[str, Any] | None = ...,
        secrets: dict[str, Any] | None = ...,
        flavor: SpaceHardware | None = ...,
        timeout: float | str | None = ...,
        namespace: str | None = ...,
        token: bool | str | None = ...,
    ) -> ScheduledJobInfo: ...
    def list_scheduled_jobs(
        self, *, timeout: int | None = ..., namespace: str | None = ..., token: bool | str | None = ...
    ) -> list[ScheduledJobInfo]: ...
    def inspect_scheduled_job(
        self, *, scheduled_job_id: str, namespace: str | None = ..., token: bool | str | None = ...
    ) -> ScheduledJobInfo: ...
    def delete_scheduled_job(
        self, *, scheduled_job_id: str, namespace: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    def suspend_scheduled_job(
        self, *, scheduled_job_id: str, namespace: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    def resume_scheduled_job(
        self, *, scheduled_job_id: str, namespace: str | None = ..., token: bool | str | None = ...
    ) -> None: ...
    @experimental
    def create_scheduled_uv_job(
        self,
        script: str,
        *,
        script_args: list[str] | None = ...,
        schedule: str,
        suspend: bool | None = ...,
        concurrency: bool | None = ...,
        dependencies: list[str] | None = ...,
        python: str | None = ...,
        image: str | None = ...,
        env: dict[str, Any] | None = ...,
        secrets: dict[str, Any] | None = ...,
        flavor: SpaceHardware | None = ...,
        timeout: float | str | None = ...,
        namespace: str | None = ...,
        token: bool | str | None = ...,
        _repo: str | None = ...,
    ) -> ScheduledJobInfo: ...

api = ...
whoami = ...
auth_check = ...
get_token_permission = ...
list_models = ...
model_info = ...
list_datasets = ...
dataset_info = ...
list_spaces = ...
space_info = ...
list_papers = ...
paper_info = ...
repo_exists = ...
revision_exists = ...
file_exists = ...
repo_info = ...
list_repo_files = ...
list_repo_refs = ...
list_repo_commits = ...
list_repo_tree = ...
get_paths_info = ...
get_model_tags = ...
get_dataset_tags = ...
create_commit = ...
create_repo = ...
delete_repo = ...
update_repo_visibility = ...
update_repo_settings = ...
move_repo = ...
upload_file = ...
upload_folder = ...
delete_file = ...
delete_folder = ...
delete_files = ...
upload_large_folder = ...
preupload_lfs_files = ...
create_branch = ...
delete_branch = ...
create_tag = ...
delete_tag = ...
get_full_repo_name = ...
super_squash_history = ...
list_lfs_files = ...
permanently_delete_lfs_files = ...
get_safetensors_metadata = ...
parse_safetensors_file_metadata = ...
run_as_future = ...
list_liked_repos = ...
list_repo_likers = ...
unlike = ...
get_discussion_details = ...
get_repo_discussions = ...
create_discussion = ...
create_pull_request = ...
change_discussion_status = ...
comment_discussion = ...
edit_discussion_comment = ...
rename_discussion = ...
merge_pull_request = ...
add_space_secret = ...
delete_space_secret = ...
get_space_variables = ...
add_space_variable = ...
delete_space_variable = ...
get_space_runtime = ...
request_space_hardware = ...
set_space_sleep_time = ...
pause_space = ...
restart_space = ...
duplicate_space = ...
request_space_storage = ...
delete_space_storage = ...
list_inference_endpoints = ...
create_inference_endpoint = ...
get_inference_endpoint = ...
update_inference_endpoint = ...
delete_inference_endpoint = ...
pause_inference_endpoint = ...
resume_inference_endpoint = ...
scale_to_zero_inference_endpoint = ...
create_inference_endpoint_from_catalog = ...
list_inference_catalog = ...
get_collection = ...
list_collections = ...
create_collection = ...
update_collection_metadata = ...
delete_collection = ...
add_collection_item = ...
update_collection_item = ...
delete_collection_item = ...
delete_collection_item = ...
list_pending_access_requests = ...
list_accepted_access_requests = ...
list_rejected_access_requests = ...
cancel_access_request = ...
accept_access_request = ...
reject_access_request = ...
grant_access = ...
create_webhook = ...
disable_webhook = ...
delete_webhook = ...
enable_webhook = ...
get_webhook = ...
list_webhooks = ...
update_webhook = ...
get_user_overview = ...
get_organization_overview = ...
list_organization_members = ...
list_user_followers = ...
list_user_following = ...
run_job = ...
fetch_job_logs = ...
list_jobs = ...
inspect_job = ...
cancel_job = ...
run_uv_job = ...
create_scheduled_job = ...
list_scheduled_jobs = ...
inspect_scheduled_job = ...
delete_scheduled_job = ...
suspend_scheduled_job = ...
resume_scheduled_job = ...
create_scheduled_uv_job = ...
