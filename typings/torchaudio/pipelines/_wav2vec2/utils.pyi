from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchaudio.models import Wav2Vec2Model

class _Wav2Vec2Model(nn.Module):
    def __init__(
        self, model: Wav2Vec2Model, normalize_waveform: bool, apply_log_softmax: bool, append_star: bool
    ) -> None: ...
    def forward(self, waveforms: Tensor, lengths: Tensor | None = ...) -> tuple[Tensor, Tensor | None]: ...
    @torch.jit.export
    def extract_features(
        self, waveforms: Tensor, lengths: Tensor | None = ..., num_layers: int | None = ...
    ) -> tuple[list[Tensor], Tensor | None]: ...
