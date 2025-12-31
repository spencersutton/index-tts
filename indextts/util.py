import functools
from collections.abc import Callable
from typing import Any

from torch import nn


def patch_call[**P, R](_src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    @functools.wraps(_src_func)
    def _returns_nn_module_call(*_args: object) -> Any:
        return nn.Module.__call__

    return _returns_nn_module_call
