from typing import Any, Callable, TypeVar, TypeAlias

FuncType: TypeAlias = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

def context_decorator(
    ctx, func
):  # -> _Wrapped[..., Any, ..., Generator[Any, Any, Any]] | _Wrapped[..., Any, ..., Any]:

    ...

class _DecoratorContextManager:
    def __call__(self, orig_func: F) -> F: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    def clone(self):  # -> Self:
        ...

class _NoParamDecoratorContextManager(_DecoratorContextManager):
    def __new__(cls, orig_func=...):  # -> Self:
        ...
