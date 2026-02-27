import re
import site
import sys
from pathlib import Path
from types import TracebackType

import rich.traceback

sitepackage = Path(site.getsitepackages()[0])
suppressed = ["transformers", "torch", "beartype"]
suppressed = [str(sitepackage / x) for x in suppressed]

rich.traceback.install(suppress=suppressed, width=None)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")

_rich_excepthook = sys.excepthook


def _excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType) -> None:
    if exc_value.args:
        exc_value.args = tuple(_ANSI_ESCAPE.sub("", arg) if isinstance(arg, str) else arg for arg in exc_value.args)  # pyright: ignore[reportAny]
    _rich_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook
