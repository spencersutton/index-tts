from typing import Optional, TypeVar

"""Miscellaneous utilities to aid with typing."""
T = TypeVar("T")

def not_none(obj: T | None) -> T: ...
