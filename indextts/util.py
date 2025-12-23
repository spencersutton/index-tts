import types
import typing
from collections.abc import Callable
from functools import cache, wraps
from pathlib import Path
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

import torch
from torch import nn


def _returns_nn_module_call(*_args: object):  # noqa: ANN202
    return nn.Module.__call__


def patch_call[**P, R](_src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    return _returns_nn_module_call


shape_file = Path("shapes.log")


def _extract_tensor_shape_spec(tp: Any) -> tuple[bool, bool, tuple[Any, ...]] | None:
    """Extract (is_list, is_optional, shape_tuple) from supported annotations.

    Supported forms:
    - Annotated[torch.Tensor, (<dims...>)]
    - list[Annotated[torch.Tensor, (<dims...>)]]
    - (Annotated[torch.Tensor, (<dims...>)] | None)
    - (list[Annotated[torch.Tensor, (<dims...>)]] | None)
    """

    def _inner(t: Any) -> tuple[bool, bool, tuple[Any, ...]] | None:
        origin = get_origin(t)

        # Optional / Union (including PEP604 |)
        if origin is types.UnionType or origin is typing.Union:
            args = list(get_args(t))
            is_optional = any(a is type(None) for a in args)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) != 1:
                return None
            inner = _inner(non_none[0])
            if inner is None:
                return None
            is_list, _opt, metadata = inner
            return is_list, (is_optional or _opt), metadata

        # list[T]
        if origin is list:
            (elem,) = get_args(t) or (None,)
            if elem is None:
                return None
            inner = _inner(elem)
            if inner is None:
                return None
            _is_list, is_optional, metadata = inner
            # Nested lists aren't supported; treat any list wrapper as "list".
            return True, is_optional, metadata

        if origin is Annotated:
            type_args = get_args(t)
            if len(type_args) != 2 or type_args[0] is not torch.Tensor:
                return None
            _, metadata = type_args
            if not isinstance(metadata, tuple):
                return None
            return False, False, metadata

        return None

    return _inner(tp)


def verify_shapes[F: Callable[..., Any]](func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            hints = get_type_hints(func, include_extras=True)
            for k, v in hints.items():
                if k == "return":
                    continue

                spec = _extract_tensor_shape_spec(v)
                if spec is None:
                    continue

                is_list, is_optional, metadata = spec
                expected = tuple("..." if x is ... else x for x in metadata)

                # Fetch argument value if it was actually provided.
                if k in kwargs:
                    value = kwargs[k]
                else:
                    try:
                        idx = func.__code__.co_varnames.index(k)
                    except ValueError:
                        continue
                    if idx >= len(args):
                        continue
                    value = args[idx]

                if value is None:
                    if is_optional:
                        continue
                    # Non-optional but None: nothing to validate.
                    continue

                def _check_one(t: Any, suffix: str = "", metadata=metadata, expected=expected, k=k) -> None:  # noqa: ANN001
                    if not isinstance(t, torch.Tensor):
                        return
                    if ... in metadata:
                        line = f"{func.__code__.co_filename}:{func.__code__.co_firstlineno} {func.__name__}:{k}{suffix} {t.shape}"
                        with shape_file.open("a") as f:
                            f.write(line + "\n")
                        print(line)

                    for i, dim in enumerate(metadata):
                        if dim == ...:
                            continue
                        if i >= t.ndim or t.shape[i] != dim:
                            print(
                                f"Argument {func.__name__}:{k}{suffix} has incorrect shape {t.shape}, expected {expected}"
                            )
                            return

                if is_list:
                    if not isinstance(value, list):
                        continue
                    for j, item in enumerate(value):
                        _check_one(item, suffix=f"[{j}]")
                else:
                    _check_one(value)
        except Exception as e:  # noqa: BLE001
            print(f"Shape verification failed in {func.__name__}: {e}")
        return func(*args, **kwargs)

    return cast(F, wrapper)


@cache
def print_once(*args: Any, **kwargs: Any) -> str:
    print(*args, **kwargs)
    return " ".join(str(arg) for arg in args)
