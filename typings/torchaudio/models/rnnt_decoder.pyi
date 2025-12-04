from collections.abc import Callable
from typing import List, Optional, Tuple, TypeAlias

import torch
from torchaudio.models import RNNT

__all__ = ["Hypothesis", "RNNTBeamSearch"]
Hypothesis: TypeAlias = tuple[list[int], torch.Tensor, list[list[torch.Tensor]], float]

class RNNTBeamSearch(torch.nn.Module):
    def __init__(
        self,
        model: RNNT,
        blank: int,
        temperature: float = ...,
        hypo_sort_key: Callable[[Hypothesis], float] | None = ...,
        step_max_tokens: int = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor, length: torch.Tensor, beam_width: int) -> list[Hypothesis]: ...
    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        beam_width: int,
        state: list[list[torch.Tensor]] | None = ...,
        hypothesis: list[Hypothesis] | None = ...,
    ) -> tuple[list[Hypothesis], list[list[torch.Tensor]]]: ...
