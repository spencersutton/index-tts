import inspect
from typing import Any

import torch
from torch._guards import Source
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Constraint
from torch.fx.experimental.symbolic_shapes import EqualityConstraint
from torch.utils._pytree import KeyPath

log = ...

class _KeyPath:
    def __init__(self, kp: KeyPath) -> None: ...

class _KeyPathTrie:
    def __init__(self) -> None: ...
    def add(self, kp: KeyPath, src: Source): ...
    def get(self, kp: KeyPath) -> tuple[Source, KeyPath]: ...

def make_sourced_prefixes(nn_module, args, kwargs) -> _KeyPathTrie: ...
def key_path_to_source(kp: KeyPath, sourced_prefixes: _KeyPathTrie | None = ...) -> Source: ...
def fakify(
    mode: FakeTensorMode,
    kp: KeyPath,
    t: Any,
    t_constraints: dict[int, dict[int, Constraint]],
    sources: dict[tuple[int, int], list[Source]],
    sourced_prefixes: _KeyPathTrie | None = ...,
): ...
def make_fake_inputs(nn_module, args, kwargs, dynamic_shapes, prefer_deferred_runtime_asserts_over_guards=...): ...
def produce_guards_and_solve_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    equalities_inputs: EqualityConstraint,
    original_signature: inspect.Signature,
): ...
def is_int(x: object) -> bool: ...
def make_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    combined_args: dict[str, Any],
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    num_lifted_inputs: int,
): ...

class _NonStrictTorchFunctionHandler(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=..., kwargs=...): ...
