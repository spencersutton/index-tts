import site
from pathlib import Path

import rich.traceback

sitepackage = Path(site.getsitepackages()[0])
suppressed = ["transformers", "torch", "beartype"]
suppressed = [str(sitepackage / x) for x in suppressed]

rich.traceback.install(suppress=suppressed, width=None)
