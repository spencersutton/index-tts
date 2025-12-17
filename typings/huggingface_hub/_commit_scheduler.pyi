from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Self

from .hf_api import CommitInfo, HfApi

logger = ...

@dataclass(frozen=True)
class _FileToUpload:
    local_path: Path
    path_in_repo: str
    size_limit: int
    last_modified: float

class CommitScheduler:
    def __init__(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        every: float = ...,
        path_in_repo: str | None = ...,
        repo_type: str | None = ...,
        revision: str | None = ...,
        private: bool | None = ...,
        token: str | None = ...,
        allow_patterns: list[str] | str | None = ...,
        ignore_patterns: list[str] | str | None = ...,
        squash_history: bool = ...,
        hf_api: HfApi | None = ...,
    ) -> None: ...
    def stop(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def trigger(self) -> Future: ...
    def push_to_hub(self) -> CommitInfo | None: ...

class PartialFileIO(BytesIO):
    def __init__(self, file_path: str | Path, size_limit: int) -> None: ...
    def __del__(self) -> None: ...
    def __len__(self) -> int: ...
    def __getattribute__(self, name: str):  # -> Any:
        ...
    def tell(self) -> int: ...
    def seek(self, __offset: int, __whence: int = ...) -> int: ...
    def read(self, __size: int | None = ...) -> bytes: ...
