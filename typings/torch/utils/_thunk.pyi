from collections.abc import Callable
from typing import TypeVar

R = TypeVar("R")

class Thunk[R]:
    """
    A simple lazy evaluation implementation that lets you delay
    execution of a function.  It properly handles releasing the
    function once it is forced.
    """

    f: Callable[[], R] | None
    r: R | None
    __slots__ = ...
    def __init__(self, f: Callable[[], R]) -> None: ...
    def force(self) -> R: ...
