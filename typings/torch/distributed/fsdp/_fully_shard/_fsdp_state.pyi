import torch
from torch import nn
from torch.distributed._composable_state import _State

from ._fsdp_api import MixedPrecisionPolicy

logger = ...

class FSDPStateContext:
    """This has state shared across FSDP states."""
    def __init__(self) -> None: ...

def disable_if_config_true(func): ...

class FSDPState(_State):
    def __init__(self) -> None: ...
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None: ...
