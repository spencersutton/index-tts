from collections.abc import Callable

from torch import nn


def patch_call[**P, R](src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    return nn.Module.__call__
