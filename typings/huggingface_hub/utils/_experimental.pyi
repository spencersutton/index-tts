from collections.abc import Callable

"""Contains utilities to flag a feature as "experimental" in Huggingface Hub."""

def experimental(fn: Callable) -> Callable: ...
