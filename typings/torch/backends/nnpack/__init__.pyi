from contextlib import contextmanager

__all__ = ["flags", "is_available", "set_flags"]

def is_available() -> bool:
    """Return whether PyTorch is built with NNPACK support."""

def set_flags(_enabled) -> tuple[bool]:
    """Set if nnpack is enabled globally"""

@contextmanager
def flags(enabled=...) -> Generator[None, Any, None]:
    """Context manager for setting if nnpack is enabled globally"""
