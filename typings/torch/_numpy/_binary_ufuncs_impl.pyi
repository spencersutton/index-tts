"""
Export torch work functions for binary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch._numpy/_ufuncs.py` module.
"""

def matmul(x, y): ...
def divmod(x, y): ...
