from pathlib import Path
from typing import Literal

from tqdm.auto import tqdm as base_tqdm

from .utils import validate_hf_hub_args

logger = ...
VERY_LARGE_REPO_THRESHOLD = ...

@validate_hf_hub_args
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = ...,
    revision: str | None = ...,
    cache_dir: str | Path | None = ...,
    local_dir: str | Path | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    user_agent: dict | str | None = ...,
    proxies: dict | None = ...,
    etag_timeout: float = ...,
    force_download: bool = ...,
    token: bool | str | None = ...,
    local_files_only: bool = ...,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    max_workers: int = ...,
    tqdm_class: type[base_tqdm] | None = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
    local_dir_use_symlinks: bool | Literal["auto"] = ...,
    resume_download: bool | None = ...,
) -> str: ...
