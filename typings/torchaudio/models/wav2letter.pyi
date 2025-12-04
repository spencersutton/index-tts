from torch import Tensor, nn

__all__ = ["Wav2Letter"]

class Wav2Letter(nn.Module):
    def __init__(self, num_classes: int = ..., input_type: str = ..., num_features: int = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
