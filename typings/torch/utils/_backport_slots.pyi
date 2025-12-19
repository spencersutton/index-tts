from typing import TypeVar

from _typeshed import DataclassInstance

__all__ = ["dataclass_slots"]
_T = TypeVar("_T", bound=DataclassInstance)

def dataclass_slots[T: DataclassInstance](cls: type[_T]) -> type[DataclassInstance]: ...
