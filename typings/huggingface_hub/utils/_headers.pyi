from ._deprecation import _deprecate_arguments
from ._validators import validate_hf_hub_args

"""Contains utilities to handle headers to send in calls to Huggingface Hub."""

@_deprecate_arguments(
    version="1.0",
    deprecated_args="is_write_action",
    custom_message=...,
)
@validate_hf_hub_args
def build_hf_headers(
    *,
    token: bool | str | None = ...,
    library_name: str | None = ...,
    library_version: str | None = ...,
    user_agent: dict | str | None = ...,
    headers: dict[str, str] | None = ...,
    is_write_action: bool = ...,
) -> dict[str, str]: ...
def get_token_to_send(token: bool | str | None) -> str | None: ...
