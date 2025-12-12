__all__ = ["custom_op", "impl", "impl_abstract", "get_ctx", "impl_save_for_backward", "impl_backward"]

def custom_op(qualname, func_or_schema=...):  # -> Callable[..., FunctionType] | FunctionType | None:

    ...
def impl(qualname, *, device_types=..., func=...):  # -> Callable[..., Any]:

    ...
def impl_abstract(qualname, *, func=...):  # -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:

    ...
def impl_save_for_backward(qualname, *, func=...):  # -> Callable[..., Any]:

    ...
def impl_backward(qualname, output_differentiability=..., *, func=...):  # -> Callable[..., Any]:

    ...
