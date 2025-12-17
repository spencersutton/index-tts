from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

class SegmentedTree[T]:
    def __init__(
        self, values: list[T], update_op: Callable[[T, T], T], summary_op: Callable[[T, T], T], identity_element: T
    ) -> None: ...
    def update_range(self, start: int, end: int, value: T) -> None: ...
    def summarize_range(self, start: int, end: int) -> T: ...
