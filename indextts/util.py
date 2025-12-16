from collections.abc import Callable

from torch import nn


def _returns_nn_module_call(*_args: object):  # noqa: ANN202
    return nn.Module.__call__


def patch_call[**P, R](_src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    return _returns_nn_module_call
