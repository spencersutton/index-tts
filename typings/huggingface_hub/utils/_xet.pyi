from dataclasses import dataclass
from enum import StrEnum

import requests

from . import validate_hf_hub_args

class XetTokenType(StrEnum):
    READ = ...
    WRITE = ...

@dataclass(frozen=True)
class XetFileData:
    file_hash: str
    refresh_route: str

@dataclass(frozen=True)
class XetConnectionInfo:
    access_token: str
    expiration_unix_epoch: int
    endpoint: str

def parse_xet_file_data_from_response(
    response: requests.Response, endpoint: str | None = ...
) -> XetFileData | None: ...
def parse_xet_connection_info_from_headers(headers: dict[str, str]) -> XetConnectionInfo | None: ...
@validate_hf_hub_args
def refresh_xet_connection_info(*, file_data: XetFileData, headers: dict[str, str]) -> XetConnectionInfo: ...
@validate_hf_hub_args
def fetch_xet_connection_info_from_repo_info(
    *,
    token_type: XetTokenType,
    repo_id: str,
    repo_type: str,
    revision: str | None = ...,
    headers: dict[str, str],
    endpoint: str | None = ...,
    params: dict[str, str] | None = ...,
) -> XetConnectionInfo: ...
