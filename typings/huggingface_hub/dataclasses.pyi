from collections.abc import Callable
from dataclasses import _MISSING_TYPE
from typing import Any, TypeVar, overload

type Validator_T = Callable[[Any], None]
T = TypeVar("T")

@overload
def strict[T](cls: type[T]) -> type[T]: ...
@overload
def strict(*, accept_kwargs: bool = ...) -> Callable[[type[T]], type[T]]: ...
def strict[T](cls: type[T] | None = ..., *, accept_kwargs: bool = ...) -> type[T] | Callable[[type[T]], type[T]]: ...
def validated_field(
    validator: list[Validator_T] | Validator_T,
    default: Any | _MISSING_TYPE = ...,
    default_factory: Callable[[], Any] | _MISSING_TYPE = ...,
    init: bool = ...,
    repr: bool = ...,
    hash: bool | None = ...,
    compare: bool = ...,
    metadata: dict | None = ...,
    **kwargs: Any,
) -> Any: ...
def as_validated_field(validator: Validator_T):  # -> Callable[..., Any]:

    ...
def type_validator(name: str, value: Any, expected_type: Any) -> None: ...

_BASIC_TYPE_VALIDATORS = ...
__all__ = [
    "StrictDataclassClassValidationError",
    "StrictDataclassDefinitionError",
    "StrictDataclassFieldValidationError",
    "Validator_T",
    "strict",
    "validated_field",
]
