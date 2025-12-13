from collections import OrderedDict
from typing import TypeVar
from collections.abc import Callable
from typing import ParamSpec

simple_call_counter: OrderedDict[str, int] = ...
_P = ParamSpec("_P")
_R = TypeVar("_R")

def count_label(label: str) -> None: ...
def count[**P, R](fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
