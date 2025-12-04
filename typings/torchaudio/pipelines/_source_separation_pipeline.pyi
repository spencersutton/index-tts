from collections.abc import Callable
from dataclasses import dataclass

import torch

@dataclass
class SourceSeparationBundle:
    _model_path: str
    _model_factory_func: Callable[[], torch.nn.Module]
    _sample_rate: int
    @property
    def sample_rate(self) -> int: ...
    def get_model(self) -> torch.nn.Module: ...

CONVTASNET_BASE_LIBRI2MIX = ...
HDEMUCS_HIGH_MUSDB_PLUS = ...
HDEMUCS_HIGH_MUSDB = ...
