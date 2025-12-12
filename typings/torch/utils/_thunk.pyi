from typing import Callable, Generic, Optional, TypeVar

R = TypeVar("R")

class Thunk(Generic[R]):
    f: Optional[Callable[[], R]]
    r: Optional[R]
    __slots__ = ...
    def __init__(self, f: Callable[[], R]) -> None: ...
    def force(self) -> R: ...
