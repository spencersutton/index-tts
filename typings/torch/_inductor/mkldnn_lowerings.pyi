import torch
from typing import Optional, Union
from . import ir
from .ir import TensorBox
from .virtualized import OpsValue

def create_int8_compensation(
    W_tensor: torch.Tensor,
    packed_weight: ir.TensorBox,
    x_scale: ir.TensorBox,
    x_zp: ir.TensorBox,
    w_scale: ir.TensorBox,
) -> tuple[
    bool,
    Union[ir.TensorBox, ir.ShapeAsConstantBuffer],
    Optional[Union[ir.TensorBox, ir.ShapeAsConstantBuffer]],
]: ...
def codegen_int8_gemm_template_compensation(
    use_int8_fast_compensation_path: bool,
    input: OpsValue,
    _weight_compo: OpsValue,
    _x_scale: Optional[OpsValue],
    _x_zp: Optional[OpsValue],
    _w_scale: Optional[OpsValue],
    _x_w_scale: Optional[OpsValue],
) -> OpsValue: ...
def grouped_gemm_lowering(
    x: TensorBox, w: list[TensorBox], b: list[TensorBox], attr=..., scalars=..., algorithm=..., layout=...
):  # -> list[TensorBox | ShapeAsConstantBuffer]:
    ...
def register_onednn_fusion_ops():  # -> None:
    ...
