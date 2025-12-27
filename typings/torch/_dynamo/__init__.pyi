"""
TorchDynamo is a Python-level JIT compiler designed to make unmodified PyTorch programs faster.
TorchDynamo hooks into the frame evaluation API in CPython (PEP 523) to dynamically modify Python
bytecode right before it is executed. It rewrites Python bytecode in order to extract sequences of
PyTorch operations into an FX Graph which is then just-in-time compiled with a customizable backend.
It creates this FX Graph through bytecode analysis and is designed to mix Python execution with
compiled backends to get the best of both worlds: usability and performance. This allows it to
seamlessly optimize PyTorch programs, including those using modern Python features.
"""

import torch

from . import config
from .backends.registry import list_backends, lookup_backend, register_backend
from .convert_frame import replay
from .decorators import (
    allow_in_graph,
    assume_constant_result,
    disable,
    disallow_in_graph,
    dont_skip_tracing,
    error_on_graph_break,
    forbid_in_graph,
    graph_break,
    mark_dynamic,
    mark_static,
    mark_static_address,
    maybe_mark_dynamic,
    nonstrict_trace,
    patch_dynamo_config,
    run,
    set_stance,
    skip_frame,
    substitute_in_graph,
)
from .eval_frame import OptimizedModule, explain, export, optimize, optimize_assert
from .external_utils import is_compiling

__all__ = [
    "OptimizedModule",
    "allow_in_graph",
    "assume_constant_result",
    "config",
    "disable",
    "disallow_in_graph",
    "dont_skip_tracing",
    "error_on_graph_break",
    "explain",
    "export",
    "forbid_in_graph",
    "graph_break",
    "is_compiling",
    "list_backends",
    "lookup_backend",
    "mark_dynamic",
    "mark_static",
    "mark_static_address",
    "maybe_mark_dynamic",
    "nonstrict_trace",
    "optimize",
    "optimize_assert",
    "patch_dynamo_config",
    "register_backend",
    "replay",
    "reset",
    "run",
    "set_stance",
    "skip_frame",
    "substitute_in_graph",
]
if torch.manual_seed is torch.random.manual_seed: ...

def reset() -> None:
    """
    Clear all compile caches and restore initial state.  This function is intended
    to reset Dynamo's state *as if* you had started a fresh process invocation, which
    makes it good for testing scenarios where you want to behave as if you started
    a new process.  It does NOT affect any file system caches.

    NB: this does NOT reset logging state.  Don't use this to test logging
    initialization/reinitialization.
    """

def reset_code_caches() -> None:
    """
    Clears in-memory code cache, which is what stores compiled products.  This
    resets less state than :func:`reset` and is mostly only used for testing
    purposes.
    """
