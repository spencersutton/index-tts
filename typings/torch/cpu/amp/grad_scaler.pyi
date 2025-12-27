from warnings import deprecated

import torch

__all__ = ["GradScaler"]

class GradScaler(torch.amp.GradScaler):
    """
    See :class:`torch.amp.GradScaler`.
    ``torch.cpu.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cpu", args...)`` instead.
    """
    @deprecated(
        "`torch.cpu.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cpu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = ...,
        growth_factor: float = ...,
        backoff_factor: float = ...,
        growth_interval: int = ...,
        enabled: bool = ...,
    ) -> None: ...
