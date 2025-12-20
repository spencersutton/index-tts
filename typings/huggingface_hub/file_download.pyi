from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from .utils import XetFileData, tqdm, validate_hf_hub_args

logger = ...
_CACHED_NO_EXIST = ...
type _CACHED_NO_EXIST_T = Any
HEADER_FILENAME_PATTERN = ...
REGEX_COMMIT_HASH = ...
REGEX_SHA256 = ...
_are_symlinks_supported_in_dir: dict[str, bool] = ...

def are_symlinks_supported(cache_dir: str | Path | None = ...) -> bool: ...

@dataclass(frozen=True)
class HfFileMetadata:
    commit_hash: str | None
    etag: str | None
    location: str
    size: int | None
    xet_file_data: XetFileData | None

def hf_hub_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = ...,
    repo_type: str | None = ...,
    revision: str | None = ...,
    endpoint: str | None = ...,
) -> str: ...
def http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    proxies: dict | None = ...,
    resume_size: int = ...,
    headers: dict[str, Any] | None = ...,
    expected_size: int | None = ...,
    displayed_filename: str | None = ...,
    _nb_retries: int = ...,
    _tqdm_bar: tqdm | None = ...,
) -> None: ...
def xet_get(
    *,
    incomplete_path: Path,
    xet_file_data: XetFileData,
    headers: dict[str, str],
    expected_size: int | None = ...,
    displayed_filename: str | None = ...,
    _tqdm_bar: tqdm | None = ...,
) -> None: ...
def repo_folder_name(*, repo_id: str, repo_type: str) -> str: ...
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = ...,
    repo_type: str | None = ...,
    revision: str | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    cache_dir: str | Path | None = ...,
    local_dir: str | Path | None = ...,
    user_agent: dict[str, str] | str | None = ...,
    force_download: bool = ...,
    proxies: dict[str, str] | None = ...,
    etag_timeout: float = ...,
    token: bool | str | None = ...,
    local_files_only: bool = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
    resume_download: bool | None = ...,
    force_filename: str | None = ...,
    local_dir_use_symlinks: bool | Literal["auto"] = ...,
) -> str: ...
def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: str | Path | None = ...,
    revision: str | None = ...,
    repo_type: str | None = ...,
) -> str | _CACHED_NO_EXIST_T | None: ...
def get_hf_file_metadata(
    url: str,
    token: bool | str | None = ...,
    proxies: dict | None = ...,
    timeout: float | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    user_agent: dict | str | None = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
) -> HfFileMetadata: ...
