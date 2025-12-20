from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, TypedDict

from ._commit_api import CommitOperationAdd
from .utils import validate_hf_hub_args

"""Git LFS related type definitions and utilities"""
if TYPE_CHECKING: ...
logger = ...
OID_REGEX = ...
LFS_MULTIPART_UPLOAD_COMMAND = ...
LFS_HEADERS = ...

@dataclass
class UploadInfo:
    sha256: bytes
    size: int
    sample: bytes
    @classmethod
    def from_path(cls, path: str):  # -> Self:
        ...
    @classmethod
    def from_bytes(cls, data: bytes):  # -> Self:
        ...
    @classmethod
    def from_fileobj(cls, fileobj: BinaryIO):  # -> Self:
        ...

@validate_hf_hub_args
def post_lfs_batch_info(
    upload_infos: Iterable[UploadInfo],
    token: str | None,
    repo_type: str,
    repo_id: str,
    revision: str | None = ...,
    endpoint: str | None = ...,
    headers: dict[str, str] | None = ...,
    transfers: list[str] | None = ...,
) -> tuple[list[dict], list[dict], str | None]: ...

class PayloadPartT(TypedDict):
    partNumber: int
    etag: str

class CompletionPayloadT(TypedDict):
    oid: str
    parts: list[PayloadPartT]

def lfs_upload(
    operation: CommitOperationAdd,
    lfs_batch_action: dict,
    token: str | None = ...,
    headers: dict[str, str] | None = ...,
    endpoint: str | None = ...,
) -> None: ...
