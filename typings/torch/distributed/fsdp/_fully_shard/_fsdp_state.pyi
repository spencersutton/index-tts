from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State

from ._fsdp_api import MixedPrecisionPolicy

if TYPE_CHECKING: ...
logger = ...

class FSDPStateContext:
    def __init__(self) -> None: ...

def disable_if_config_true(func):  # -> _Wrapped[..., Any, ..., Any]:
    ...

class FSDPState(_State):
    def __init__(self) -> None: ...
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None: ...
