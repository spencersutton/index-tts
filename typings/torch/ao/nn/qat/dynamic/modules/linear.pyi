from typing import TYPE_CHECKING

import torch
from torch.ao.quantization.qconfig import QConfig

if TYPE_CHECKING: ...
__all__ = ["Linear"]

class Linear(torch.ao.nn.qat.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        qconfig: QConfig | None = ...,
        device: int | str | torch.device | None = ...,
        dtype: str | None = ...,
    ) -> None: ...
