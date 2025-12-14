from typing import TYPE_CHECKING, TypeVar

from _typeshed import DataclassInstance

if TYPE_CHECKING: ...
__all__ = ["dataclass_slots"]
_T = TypeVar("_T", bound=DataclassInstance)

def dataclass_slots[T: DataclassInstance](cls: type[_T]) -> type[DataclassInstance]: ...
