from dataclasses import dataclass
from pathlib import Path

"""Contains utilities to handle the `../.cache/huggingface` folder in local directories.

First discussed in https://github.com/huggingface/huggingface_hub/issues/1738 to store
download metadata when downloading files from the hub to a local directory (without
using the cache).

./.cache/huggingface folder structure:
[4.0K]  data
├── [4.0K]  .cache
│   └── [4.0K]  huggingface
│       └── [4.0K]  download
│           ├── [  16]  file.parquet.metadata
│           ├── [  16]  file.txt.metadata
│           └── [4.0K]  folder
│               └── [  16]  file.parquet.metadata
│
├── [6.5G]  file.parquet
├── [1.5K]  file.txt
└── [4.0K]  folder
    └── [   16]  file.parquet


Download metadata file structure:
```
# file.txt.metadata
11c5a3d5811f50298f278a704980280950aedb10
a16a55fda99d2f2e7b69cce5cf93ff4ad3049930
1712656091.123

# file.parquet.metadata
11c5a3d5811f50298f278a704980280950aedb10
7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421
1712656091.123
}
```
"""
logger = ...

@dataclass
class LocalDownloadFilePaths:
    file_path: Path
    lock_path: Path
    metadata_path: Path
    def incomplete_path(self, etag: str) -> Path: ...

@dataclass(frozen=True)
class LocalUploadFilePaths:
    path_in_repo: str
    file_path: Path
    lock_path: Path
    metadata_path: Path

@dataclass
class LocalDownloadFileMetadata:
    filename: str
    commit_hash: str
    etag: str
    timestamp: float

@dataclass
class LocalUploadFileMetadata:
    size: int
    timestamp: float | None = ...
    should_ignore: bool | None = ...
    sha256: str | None = ...
    upload_mode: str | None = ...
    remote_oid: str | None = ...
    is_uploaded: bool = ...
    is_committed: bool = ...
    def save(self, paths: LocalUploadFilePaths) -> None: ...

def get_local_download_paths(local_dir: Path, filename: str) -> LocalDownloadFilePaths: ...
def get_local_upload_paths(local_dir: Path, filename: str) -> LocalUploadFilePaths: ...
def read_download_metadata(local_dir: Path, filename: str) -> LocalDownloadFileMetadata | None: ...
def read_upload_metadata(local_dir: Path, filename: str) -> LocalUploadFileMetadata: ...
def write_download_metadata(local_dir: Path, filename: str, commit_hash: str, etag: str) -> None: ...
