import torch._logging._registrations

from ._internal import (
    DEFAULT_LOGGING,
    LazyString,
    _init_logs,
    dtrace_structured,
    get_structured_logging_overhead,
    getArtifactLogger,
    hide_warnings,
    set_logs,
    trace_structured,
    warning_once,
)
