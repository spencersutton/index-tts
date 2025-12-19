from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, NoReturn

from PIL.Image import Image
from requests import HTTPError

"""Contains utilities used by both the sync and async inference clients."""
if TYPE_CHECKING: ...
UrlT = str
type PathT = str | Path
type ContentT = bytes | BinaryIO | PathT | UrlT | Image | bytearray | memoryview
TASKS_EXPECTING_IMAGES = ...
logger = ...

@dataclass
class RequestParameters:
    url: str
    task: str
    model: str | None
    json: str | dict | list | None
    data: bytes | None
    headers: dict[str, Any]

class MimeBytes(bytes):
    mime_type: str | None
    def __new__(cls, data: bytes, mime_type: str | None = ...):  # -> Self:
        ...

_UNSUPPORTED_TEXT_GENERATION_KWARGS: dict[str | None, list[str]] = ...

def raise_text_generation_error(http_error: HTTPError) -> NoReturn: ...
