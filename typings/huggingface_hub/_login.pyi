from .utils._deprecation import _deprecate_arguments, _deprecate_positional_args

"""Contains methods to log in to the Hub."""
logger = ...
_HF_LOGO_ASCII = ...

@_deprecate_arguments(
    version="1.0",
    deprecated_args="write_permission",
    custom_message=...,
)
@_deprecate_positional_args(version="1.0")
def login(
    token: str | None = ...,
    *,
    add_to_git_credential: bool = ...,
    new_session: bool = ...,
    write_permission: bool = ...,
) -> None: ...
def logout(token_name: str | None = ...) -> None: ...
def auth_switch(token_name: str, add_to_git_credential: bool = ...) -> None: ...
def auth_list() -> None: ...
@_deprecate_arguments(
    version="1.0",
    deprecated_args="write_permission",
    custom_message=...,
)
@_deprecate_positional_args(version="1.0")
def interpreter_login(*, new_session: bool = ..., write_permission: bool = ...) -> None: ...

NOTEBOOK_LOGIN_PASSWORD_HTML = ...
NOTEBOOK_LOGIN_TOKEN_HTML_START = ...
NOTEBOOK_LOGIN_TOKEN_HTML_END = ...

@_deprecate_arguments(
    version="1.0",
    deprecated_args="write_permission",
    custom_message=...,
)
@_deprecate_positional_args(version="1.0")
def notebook_login(*, new_session: bool = ..., write_permission: bool = ...) -> None: ...
