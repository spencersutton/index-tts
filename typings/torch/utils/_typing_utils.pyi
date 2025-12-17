from typing import TypeVar

T = TypeVar("T")

def not_none[T](obj: T | None) -> T: ...
