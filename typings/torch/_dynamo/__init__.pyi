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

def reset() -> None: ...
def reset_code_caches() -> None: ...
