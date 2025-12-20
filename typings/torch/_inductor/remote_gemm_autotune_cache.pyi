import asyncio
from typing import TypeVar

from torch._inductor import ir

_T = TypeVar("_T")

def gen_best_config(mat1: ir.StorageBox, mat2: ir.StorageBox) -> asyncio.Task[_T]: ...
