"""Triton Implementation of the flex_attention Kernel"""

from dataclasses import dataclass

import torch

from ...ir import ComputedBuffer, TensorBox
from ...lowering import register_lowering
from ...select_algorithm import SymbolicGridFn
from .common import SubgraphResults

log = ...
aten = ...
Expr = ...

@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv): ...
def get_float32_precision(): ...

flex_attention_template = ...

@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
    query, key, value, subgraph, block_mask, scale, kernel_options, score_mod_other_buffers, mask_mod_other_buffers
):
    """
    The main lowering for the flex_attention hop
    This can currently lower to one of 3 templates:
    1. Base Triton Template
    2. Flex Decode Triton Template
    3. Cpu specific CPP template
    """

@SymbolicGridFn
def flex_attention_backward_grid(batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta, *, cdiv): ...

flex_attention_backward_template = ...

def validate_joint_graph(joint_graph: torch.fx.Graph):
    """We do some pre lowering graph checks in order to raise nicer error messages"""

@dataclass(frozen=True)
class JointOutputResult:
    """Results from processing joint outputs."""

    grad_input: ComputedBuffer
    captured_grads_compute: list[ComputedBuffer]
    captured_grads: list[TensorBox | None]
    mutated_grads: list[TensorBox]

def process_joint_outputs(all_joint_outputs: SubgraphResults, num_placeholders: int) -> JointOutputResult:
    """
    Process joint outputs and extract various buffers needed for lowering

    Args:
        all_joint_outputs: List of all the outputs from build_subgraphs
        num_placeholders: The number of placeholder inputs, used to skip over unused backward compute buffers

    Returns:
        JointOutputResult containing processed buffers and gradients
    """

@register_lowering(torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None)
def flex_attention_backward(*args, **kwargs):
    """Lowering for the flex_attention_backward op in triton"""

def get_bwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults, mask_graph_buffer: SubgraphResults, joint_outputs: JointOutputResult
) -> list[ComputedBuffer | TensorBox | None]: ...
