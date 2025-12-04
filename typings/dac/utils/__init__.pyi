from pathlib import Path

import argbind
import dac
from audiotools import ml

DAC = dac.model.DAC
Accelerator = ml.Accelerator
__MODEL_LATEST_TAGS__ = ...
__MODEL_URLS__ = ...

@argbind.bind(group="download", positional=True, without_prefix=True)
def download(model_type: str = ..., model_bitrate: str = ..., tag: str = ...):  # -> Path:
    ...
def load_model(model_type: str = ..., model_bitrate: str = ..., tag: str = ..., load_path: str = ...):  # -> DAC:
    ...
