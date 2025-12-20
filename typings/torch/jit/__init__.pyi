from typing import Any

import torch._C
from torch._jit_internal import Future, export, ignore, unused
from torch.jit._async import fork, wait
from torch.jit._freeze import freeze, optimize_for_inference
from torch.jit._fuser import set_fusion_strategy
from torch.jit._script import Attribute, CompilationUnit, ScriptFunction, ScriptModule, interface, script
from torch.jit._serialization import load, save
from torch.jit._trace import trace, trace_module

__all__ = [
    "Attribute",
    "CompilationUnit",
    "Error",
    "Future",
    "ScriptFunction",
    "ScriptModule",
    "annotate",
    "enable_onednn_fusion",
    "export",
    "export_opnames",
    "fork",
    "freeze",
    "ignore",
    "interface",
    "isinstance",
    "load",
    "onednn_fusion_enabled",
    "optimize_for_inference",
    "save",
    "script",
    "script_if_tracing",
    "set_fusion_strategy",
    "strict_fusion",
    "trace",
    "trace_module",
    "unused",
    "wait",
]
_fork = ...
_wait = ...
_set_fusion_strategy = ...

def export_opnames(m) -> list[str]: ...

Error = torch._C.JITException

def annotate(the_type, the_value): ...
def script_if_tracing(fn) -> Callable[..., Any]: ...
def isinstance(obj, target_type) -> bool: ...

class strict_fusion:
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, tb: Any) -> None: ...

def enable_onednn_fusion(enabled: bool) -> None: ...
def onednn_fusion_enabled() -> bool: ...

if not torch._C._jit_init(): ...
