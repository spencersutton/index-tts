from collections.abc import Iterable
from typing import TypeVar

"""Contains a utility to iterate by chunks over an iterator."""
T = TypeVar("T")

def chunk_iterable[T](iterable: Iterable[T], chunk_size: int) -> Iterable[Iterable[T]]: ...
