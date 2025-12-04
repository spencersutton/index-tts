# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

from collections import UserDict

class AttrDict(UserDict):
    def __init__(self, *args, **kwargs) -> None: ...

def build_env(config, config_name, path) -> None: ...
