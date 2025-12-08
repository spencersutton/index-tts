from collections.abc import Callable

from torch import nn


def _returns_nn_module_call(*args):
    return nn.Module.__call__


def patch_call[**P, R](
    src_func: Callable[P, R],
    return_type: type[R],
) -> Callable[..., Callable[P, R]]:
    return _returns_nn_module_call
