from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import fastapi
import gradio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from gradio.data_classes import DeveloperPath, UserProvidedPath
from gradio.i18n import I18n
from gradio.themes import ThemeClass as Theme
from gradio_client.documentation import document

"""Implements a FastAPI server to run the gradio interface. Note that some types in this
module use the Optional/Union notation so that they work correctly with pydantic."""

STATIC_TEMPLATE_LIB = ...
STATIC_PATH_LIB = ...
BUILD_PATH_LIB = ...
VERSION = ...
XSS_SAFE_MIMETYPES = ...
DEFAULT_TEMP_DIR = ...
BUILT_IN_THEMES: dict[str, Theme] = ...

class ORJSONResponse(JSONResponse):
    media_type = ...
    @staticmethod
    def default(content: Any) -> str: ...
    def render(self, content: Any) -> bytes: ...

def toorjson(value):  # -> Markup:
    ...

templates = ...
client = ...
file_upload_statuses = ...

class App(FastAPI):
    app_port = ...
    def __init__(self, auth_dependency: Callable[[fastapi.Request], str | None] | None = ..., **kwargs) -> None: ...

    client = ...
    @staticmethod
    async def proxy_to_node(
        request: fastapi.Request,
        server_name: str,
        node_port: int,
        python_port: int,
        scheme: str = ...,
        mounted_path: str = ...,
    ) -> Response: ...
    def configure_app(self, blocks: gradio.Blocks) -> None: ...
    def get_blocks(self) -> gradio.Blocks: ...
    def build_proxy_request(self, url_path):  # -> Request:
        ...
    @staticmethod
    def setup_mcp_server(
        blocks: gradio.Blocks, app_kwargs: dict[str, Any], mcp_server: bool | None = ...
    ):  # -> Literal['/gradio_api/mcp']:
        ...
    @staticmethod
    def create_app(
        blocks: gradio.Blocks,
        app_kwargs: dict[str, Any] | None = ...,
        auth_dependency: Callable[[fastapi.Request], str | None] | None = ...,
        strict_cors: bool = ...,
        ssr_mode: bool = ...,
        mcp_server: bool | None = ...,
    ) -> App: ...

def load_system_prompt(starter_queries: bool = ...):  # -> str:
    ...
def routes_safe_join(directory: DeveloperPath, path: UserProvidedPath) -> str: ...
def get_types(cls_set: list[type]):  # -> tuple[list[Any], list[Any]]:
    ...
@document()
def mount_gradio_app(
    app: fastapi.FastAPI,
    blocks: gradio.Blocks,
    path: str,
    server_name: str = ...,
    server_port: int = ...,
    footer_links: (list[Literal["api", "gradio", "settings"] | dict[str, str]] | None) = ...,
    app_kwargs: dict[str, Any] | None = ...,
    *,
    auth: Callable | tuple[str, str] | list[tuple[str, str]] | None = ...,
    auth_message: str | None = ...,
    auth_dependency: Callable[[fastapi.Request], str | None] | None = ...,
    root_path: str | None = ...,
    allowed_paths: list[str] | None = ...,
    blocked_paths: list[str] | None = ...,
    favicon_path: str | None = ...,
    show_error: bool = ...,
    max_file_size: str | int | None = ...,
    ssr_mode: bool | None = ...,
    node_server_name: str | None = ...,
    node_port: int | None = ...,
    enable_monitoring: bool | None = ...,
    pwa: bool | None = ...,
    i18n: I18n | None = ...,
    mcp_server: bool | None = ...,
    theme: Theme | str | None = ...,
    css: str | None = ...,
    css_paths: str | Path | Sequence[str | Path] | None = ...,
    js: str | Literal[True] | None = ...,
    head: str | None = ...,
    head_paths: str | Path | Sequence[str | Path] | None = ...,
) -> fastapi.FastAPI: ...

INTERNAL_ROUTES = ...
