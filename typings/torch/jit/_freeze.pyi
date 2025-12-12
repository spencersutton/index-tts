from typing import Optional
from torch.jit._script import ScriptModule

"""Freezing.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

def freeze(mod, preserved_attrs: Optional[list[str]] = ..., optimize_numerics: bool = ...):  # -> RecursiveScriptModule:

    ...
def run_frozen_optimizations(
    mod, optimize_numerics: bool = ..., preserved_methods: Optional[list[str]] = ...
):  # -> None:

    ...
def optimize_for_inference(mod: ScriptModule, other_methods: Optional[list[str]] = ...) -> ScriptModule: ...
