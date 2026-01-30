import site
from pathlib import Path

from rich.traceback import install

sitepackage = Path(site.getsitepackages()[0])
suppressed = ["transformers", "torch"]
suppressed = [str(sitepackage / x) for x in suppressed]

install(suppress=suppressed, width=None)
