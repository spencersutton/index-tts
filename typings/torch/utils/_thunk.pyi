from typing import Generic, Optional, TypeVar
from collections.abc import Callable

R = TypeVar("R")

class Thunk[R]:
    f: Callable[[], R] | None
    r: R | None
    __slots__ = ...
    def __init__(self, f: Callable[[], R]) -> None: ...
    def force(self) -> R: ...
