from os import PathLike

from torchaudio._internal.module_utils import dropping_support

_LG = ...

@dropping_support
def download_asset(key: str, hash: str = ..., path: str | PathLike = ..., *, progress: bool = ...) -> str: ...
