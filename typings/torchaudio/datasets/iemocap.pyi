from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

_SAMPLE_RATE = ...

class IEMOCAP(Dataset):
    def __init__(
        self,
        root: str | Path,
        sessions: tuple[str] = ...,
        utterance_type: str | None = ...,
    ) -> None: ...
    def get_metadata(self, n: int) -> tuple[str, int, str, str, str]: ...
    def __getitem__(self, n: int) -> tuple[Tensor, int, str, str, str]: ...
    def __len__(self) -> int: ...
