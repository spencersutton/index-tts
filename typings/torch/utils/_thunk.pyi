from collections.abc import Callable
from typing import TypeVar

R = TypeVar("R")

class Thunk[R]:
    f: Callable[[], R] | None
    r: R | None
    __slots__ = ...
    def __init__(self, f: Callable[[], R]) -> None: ...
    def force(self) -> R: ...
