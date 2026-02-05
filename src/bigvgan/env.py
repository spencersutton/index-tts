# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from collections import UserDict
from typing import Any


class AttrDict(UserDict[str, Any]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key: Any) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
