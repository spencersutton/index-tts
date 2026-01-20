# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os
import pathlib
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        pathlib.Path(path).mkdir(exist_ok=True, parents=True)
        shutil.copyfile(config, os.path.join(path, config_name))
