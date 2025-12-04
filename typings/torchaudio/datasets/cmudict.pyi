from pathlib import Path
from typing import List, Tuple, Union

from torch.utils.data import Dataset

_CHECKSUMS = ...
_PUNCTUATIONS = ...

class CMUDict(Dataset):
    def __init__(
        self,
        root: str | Path,
        exclude_punctuations: bool = ...,
        *,
        download: bool = ...,
        url: str = ...,
        url_symbols: str = ...,
    ) -> None: ...
    def __getitem__(self, n: int) -> tuple[str, list[str]]: ...
    def __len__(self) -> int: ...
    @property
    def symbols(self) -> list[str]: ...
