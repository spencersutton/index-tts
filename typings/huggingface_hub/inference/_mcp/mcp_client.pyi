from collections.abc import AsyncIterable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Unpack, overload

from typing_extensions import TypedDict

from .._generated.types import ChatCompletionInputMessage, ChatCompletionInputTool, ChatCompletionStreamOutput
from .._providers import PROVIDER_OR_POLICY_T

if TYPE_CHECKING: ...
logger = ...
type ToolName = str
type ServerType = Literal["stdio", "sse", "http"]

class StdioServerParameters_T(TypedDict):
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]
    cwd: NotRequired[str | Path | None]

class SSEServerParameters_T(TypedDict):
    url: str
    headers: NotRequired[dict[str, Any]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]

class StreamableHTTPParameters_T(TypedDict):
    url: str
    headers: NotRequired[dict[str, Any]]
    timeout: NotRequired[timedelta]
    sse_read_timeout: NotRequired[timedelta]
    terminate_on_close: NotRequired[bool]

class MCPClient:
    def __init__(
        self,
        *,
        model: str | None = ...,
        provider: PROVIDER_OR_POLICY_T | None = ...,
        base_url: str | None = ...,
        api_key: str | None = ...,
    ) -> None: ...
    async def __aenter__(self):  # -> Self:

        ...
    async def __aexit__(self, exc_type, exc_val, exc_tb):  # -> None:

        ...
    async def cleanup(self):  # -> None:

        ...
    @overload
    async def add_mcp_server(self, type: Literal["stdio"], **params: Unpack[StdioServerParameters_T]): ...
    @overload
    async def add_mcp_server(self, type: Literal["sse"], **params: Unpack[SSEServerParameters_T]): ...
    @overload
    async def add_mcp_server(self, type: Literal["http"], **params: Unpack[StreamableHTTPParameters_T]): ...
    async def add_mcp_server(self, type: ServerType, **params: Any):  # -> None:

        ...
    async def process_single_turn_with_tools(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        exit_loop_tools: list[ChatCompletionInputTool] | None = ...,
        exit_if_first_chunk_no_tool: bool = ...,
    ) -> AsyncIterable[ChatCompletionStreamOutput | ChatCompletionInputMessage]: ...
