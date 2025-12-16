# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import yaml
from torch import nn


def load_checkpoint2(model: nn.Module, model_pth: Path) -> dict[str, object]:
    checkpoint = cast(Mapping[str, object], torch.load(model_pth, map_location="cpu"))
    checkpoint = cast(Mapping[str, object], checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict=checkpoint, strict=True)
    info_path = model_pth.with_suffix(".yaml")
    configs: dict[str, object] = {}
    if info_path.exists():
        with info_path.open(encoding="utf-8") as fin:
            configs = cast(dict[str, object], yaml.safe_load(fin))
    return configs
