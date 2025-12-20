from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub.errors import CorruptedCacheException

"""Contains utilities to manage the HF cache directory."""
logger = ...
type REPO_TYPE_T = Literal["model", "dataset", "space"]
FILES_TO_IGNORE = ...

@dataclass(frozen=True)
class CachedFileInfo:
    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int
    blob_last_accessed: float
    blob_last_modified: float
    @property
    def blob_last_accessed_str(self) -> str: ...
    @property
    def blob_last_modified_str(self) -> str: ...
    @property
    def size_on_disk_str(self) -> str: ...

@dataclass(frozen=True)
class CachedRevisionInfo:
    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: frozenset[CachedFileInfo]
    refs: frozenset[str]
    last_modified: float
    @property
    def last_modified_str(self) -> str: ...
    @property
    def size_on_disk_str(self) -> str: ...
    @property
    def nb_files(self) -> int: ...

@dataclass(frozen=True)
class CachedRepoInfo:
    repo_id: str
    repo_type: REPO_TYPE_T
    repo_path: Path
    size_on_disk: int
    nb_files: int
    revisions: frozenset[CachedRevisionInfo]
    last_accessed: float
    last_modified: float
    @property
    def last_accessed_str(self) -> str: ...
    @property
    def last_modified_str(self) -> str: ...
    @property
    def size_on_disk_str(self) -> str: ...
    @property
    def refs(self) -> dict[str, CachedRevisionInfo]: ...

@dataclass(frozen=True)
class DeleteCacheStrategy:
    expected_freed_size: int
    blobs: frozenset[Path]
    refs: frozenset[Path]
    repos: frozenset[Path]
    snapshots: frozenset[Path]
    @property
    def expected_freed_size_str(self) -> str: ...
    def execute(self) -> None: ...

@dataclass(frozen=True)
class HFCacheInfo:
    size_on_disk: int
    repos: frozenset[CachedRepoInfo]
    warnings: list[CorruptedCacheException]
    @property
    def size_on_disk_str(self) -> str: ...
    def delete_revisions(self, *revisions: str) -> DeleteCacheStrategy: ...
    def export_as_table(self, *, verbosity: int = ...) -> str: ...

def scan_cache_dir(cache_dir: str | Path | None = ...) -> HFCacheInfo: ...

_TIMESINCE_CHUNKS = ...
