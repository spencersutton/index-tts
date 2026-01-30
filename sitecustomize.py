import site
from pathlib import Path

import torch
from rich.traceback import install

sitepackage = Path(site.getsitepackages()[0])
suppressed = ["transformers", "torch"]
suppressed = [str(sitepackage / x) for x in suppressed]

install(suppress=suppressed, width=None)

torch.Tensor.__repr__ = lambda self: f"Tensor(shape={tuple(self.shape)}, dtype={self.dtype})"
