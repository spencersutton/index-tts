from contextlib import contextmanager

from torch.backends import ContextProp, PropModule

def is_available() -> bool: ...

VERBOSE_OFF = ...
VERBOSE_ON = ...
VERBOSE_ON_CREATION = ...

class verbose:
    """
    On-demand oneDNN (former MKL-DNN) verbosing functionality.

    To make it easier to debug performance issues, oneDNN can dump verbose
    messages containing information like kernel size, input data size and
    execution duration while executing the kernel. The verbosing functionality
    can be invoked via an environment variable named `DNNL_VERBOSE`. However,
    this methodology dumps messages in all steps. Those are a large amount of
    verbose messages. Moreover, for investigating the performance issues,
    generally taking verbose messages for one single iteration is enough.
    This on-demand verbosing functionality makes it possible to control scope
    for verbose message dumping. In the following example, verbose messages
    will be dumped out for the second inference only.

    .. highlight:: python
    .. code-block:: python

        import torch

        model(data)
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
            - ``VERBOSE_ON_CREATION``: Enable verbosing, including oneDNN kernel creation
    """
    def __init__(self, level) -> None: ...
    def __enter__(self) -> Self | None: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]: ...

def set_flags(
    _enabled=..., _deterministic=..., _allow_tf32=..., _fp32_precision=...
) -> tuple[bool, bool, bool, str]: ...
@contextmanager
def flags(enabled=..., deterministic=..., allow_tf32=..., fp32_precision=...) -> Generator[None, Any, None]: ...

class MkldnnModule(PropModule):
    def __init__(self, m, name) -> None: ...
    def is_available(self) -> bool: ...

    enabled = ...
    deterministic = ...
    allow_tf32 = ...
    matmul = ...
    conv = ...
    rnn = ...
    fp32_precision = ...

enabled: ContextProp
deterministic: ContextProp
allow_tf32: ContextProp
