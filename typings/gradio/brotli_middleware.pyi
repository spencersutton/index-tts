from typing import NoReturn

from starlette.types import ASGIApp, Message, Receive, Scope, Send

"""AGSI Brotli middleware build on top of starlette.

Code is based on GZipMiddleware shipped with starlette.
"""

class Mode:
    generic = ...
    text = ...
    font = ...

class BrotliMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        quality: int = ...,
        mode: str = ...,
        lgwin: int = ...,
        lgblock: int = ...,
        minimum_size: int = ...,
        gzip_fallback: bool = ...,
        excluded_handlers: list[str] | None = ...,
    ) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...

class BrotliResponder:
    def __init__(self, app: ASGIApp, quality: int, mode: Mode, lgwin: int, lgblock: int, minimum_size: int) -> None: ...
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: ...
    async def send_with_brotli(self, message: Message) -> None: ...

async def unattached_send(_: Message) -> NoReturn: ...
