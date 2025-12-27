"""
This module contains pre-dispatch wrappers for functorch operations
that enable proper tracing in PT2 non-strict export/compile fx graph.
"""

DECOMPOSITIONS_LOADED = ...
DECOMPOSITIONS_LOCK = ...
VMAP_DECOMPOSITIONS_LIB = ...

def lazy_load_decompositions():
    """Lazy loading of vmap decompositions with pre-dispatch support."""
