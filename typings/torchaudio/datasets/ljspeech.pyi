from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

_RELEASE_CONFIGS = ...

class LJSPEECH(Dataset):
    def __init__(
        self,
        root: str | Path,
        url: str = ...,
        folder_in_archive: str = ...,
        download: bool = ...,
    ) -> None: ...
    def __getitem__(self, n: int) -> tuple[Tensor, int, str, str]: ...
    def __len__(self) -> int: ...
