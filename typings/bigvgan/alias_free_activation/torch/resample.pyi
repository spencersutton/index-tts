# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

from typing import Any

import torch.nn as nn
from torch import Tensor

class UpSample1d(nn.Module):
    def __init__(self, ratio=..., kernel_size=...) -> None: ...
    def forward(self, x) -> Tensor: ...

class DownSample1d(nn.Module):
    def __init__(self, ratio=..., kernel_size=...) -> None: ...
    def forward(self, x) -> Any: ...
