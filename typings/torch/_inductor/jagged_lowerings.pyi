import sympy
import torch
from typing import Union
from .ir import ShapeAsConstantBuffer, TensorBox

def dense_idx_to_jagged_idx(batch_idx, seq_idx, offsets_loader, jagged_len):  # -> tuple[Any, Any]:
    ...
def get_inverse_offsets(
    offsets: TensorBox, jagged_len: Union[int, sympy.Expr], realize: bool = ...
) -> Union[TensorBox, ShapeAsConstantBuffer]: ...
def jagged_idx_to_dense_idx(
    jagged_idx,
    inverse_offsets_loader,
    offsets_loader,
    batch_size: Union[int, sympy.Expr],
    max_seq_len: Union[int, sympy.Expr],
    offsets_dtype: torch.dtype,
) -> tuple[sympy.Expr, sympy.Expr]: ...
def register_jagged_ops():  # -> None:
    ...
