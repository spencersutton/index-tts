from typing import Any

from torch._inductor.select_algorithm import SymbolicGridFn

from ..ir import TensorBox

log = ...

@SymbolicGridFn
def mm_grid(m, n, meta, *, cdiv): ...
@SymbolicGridFn
def persistent_mm_grid(M: int, N: int, meta: dict[str, Any], *, cdiv, min): ...
@SymbolicGridFn
def persistent_grouped_mm_grid(*args): ...
def acc_type(dtype): ...
def mm_args(mat1, mat2, *others, layout=..., out_dtype=..., use_4x2_dim=..., mat2_transposed=...):
    """Common arg processing for mm,bmm,addmm,etc"""

def addmm_epilogue(dtype, alpha, beta): ...
def scale_mm_epilogue():
    """
    Create an epilogue function that applies scaling to matrix multiplication result
    using the given scale factors.

    Args:
        dtype: The data type of the output
        scale_a: Scale factor for matrix A
        scale_b: Scale factor for matrix B

    Returns:
        Epilogue function that takes the accumulator and applies scaling
    """

def check_supported_striding(mat_a: TensorBox, mat_b: TensorBox) -> None: ...
def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    """Checking if the batch stride is the largest in the stride."""
