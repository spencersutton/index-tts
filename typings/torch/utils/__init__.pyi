from torch.utils import (
    backcompat as backcompat,
    collect_env as collect_env,
    data as data,
    deterministic as deterministic,
    hooks as hooks,
)

def set_module(obj, mod):
    """Set the module attribute on a python object for a given object for nicer printing"""

cmake_prefix_path = ...

def swap_tensors(t1, t2):
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
