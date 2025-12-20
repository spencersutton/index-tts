import asyncio
from collections.abc import AsyncGenerator, Iterable

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient

from .._providers import PROVIDER_OR_POLICY_T
from .types import ServerConfig

class Agent(MCPClient):
    def __init__(
        self,
        *,
        model: str | None = ...,
        servers: Iterable[ServerConfig],
        provider: PROVIDER_OR_POLICY_T | None = ...,
        base_url: str | None = ...,
        api_key: str | None = ...,
        prompt: str | None = ...,
    ) -> None: ...
    async def load_tools(self) -> None: ...
    async def run(
        self, user_input: str, *, abort_event: asyncio.Event | None = ...
    ) -> AsyncGenerator[ChatCompletionStreamOutput | ChatCompletionInputMessage]: ...
