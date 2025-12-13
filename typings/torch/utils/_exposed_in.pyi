from typing import TypeVar
from collections.abc import Callable

F = TypeVar("F")

def exposed_in(module: str) -> Callable[[F], F]: ...
