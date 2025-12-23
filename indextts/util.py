from collections.abc import Callable
from functools import cache, wraps
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

import torch
from torch import nn


def _returns_nn_module_call(*_args: object):  # noqa: ANN202
    return nn.Module.__call__


def patch_call[**P, R](_src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    return _returns_nn_module_call


def verify_shapes[F: Callable[..., Any]](func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        hints = get_type_hints(func, include_extras=True)
        for k, v in hints.items():
            type_args = get_args(v)
            if len(type_args) != 2 or type_args[0] is not torch.Tensor or get_origin(v) != Annotated:
                continue

            _, metadata = type_args

            assert isinstance(metadata, tuple), f"Metadata for {k} must be a tuple"
            value = kwargs.get(k)
            if value is None:
                value = args[func.__code__.co_varnames.index(k)]
            for i, dim in enumerate(metadata):
                if dim == ...:
                    continue
                if value.shape[i] != dim:
                    expected = tuple("..." if x is ... else x for x in metadata)
                    raise ValueError(f"Argument {k} has incorrect shape {value.shape}, expected {expected}")

        return func(*args, **kwargs)

    return cast(F, wrapper)


@cache
def print_once(*args: Any, **kwargs: Any) -> str:
    print(*args, **kwargs)
    return " ".join(str(arg) for arg in args)
