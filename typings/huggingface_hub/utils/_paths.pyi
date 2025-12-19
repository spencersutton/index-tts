from collections.abc import Callable, Generator, Iterable
from typing import TypeVar

"""Contains utilities to handle paths in Huggingface Hub."""
T = TypeVar("T")
DEFAULT_IGNORE_PATTERNS = ...
FORBIDDEN_FOLDERS = ...

def filter_repo_objects(
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = ...,
    ignore_patterns: list[str] | str | None = ...,
    key: Callable[[T], str] | None = ...,
) -> Generator[T]: ...
