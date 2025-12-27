from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.constant_folding import ConstantFolder
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import Arg, CallFunction, KeywordArg, Match, init_once_fakemode, register_graph_pattern

log = ...
patterns = ...
aten = ...
prims = ...
pass_patterns = ...

@init_once_fakemode
def lazy_init(): ...
def remove_no_ops(gm: torch.fx.GraphModule, zeros: OrderedSet[torch.fx.Node], ones: OrderedSet[torch.fx.Node]): ...
def remove_redundant_views(gm: torch.fx.GraphModule):
    """Removes redundant views by reusing existing ones."""

class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a uniform value
    with a tensor constructor call: aten.full([shape], value, ...)
    """
    def __init__(self, gm, skip_constructors=...) -> None: ...
    def insertable_tensor_check(self, t: torch.Tensor) -> bool: ...
    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None: ...
    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None: ...

def constant_fold_uniform_value(gm: torch.fx.GraphModule): ...
def canonicalize_quant_mapping(gm: torch.fx.GraphModule):
    """
    torch.ops.higher_order.invoke_quant_packed(repeated_subgraph0, 'quant_invoke_0_0', (arg0_1, arg1_1));
    ->
    torch.ops.higher_order.invoke_quant(repeated_subgraph0, arg0_1, arg1_1, scheme = 'nf4');
    """

def canonicalize_aten_ir_passes(gm: torch.fx.GraphModule):
    """
    Canonicalization passes that will run immediately after aot autograd
    tracing. Thsis must be run before all other graph passes.
    """

def joint_graph_passes(graph: torch.fx.GraphModule):
    """Run FX transformations on the joint forwards+backwards graph."""

@register_graph_pattern(
    CallFunction(
        torch.ops.prims.iota.default,
        KeywordArg("length"),
        start=KeywordArg("start"),
        step=KeywordArg("step"),
        dtype=KeywordArg("dtype"),
        device=KeywordArg("device"),
        requires_grad=KeywordArg("requires_grad"),
    ),
    pass_dict=patterns,
)
def fix_iota_device(match: Match, length, start, step, dtype, device, requires_grad):
    """
    Eager supports:

        aten.index(cuda_tensor, torch.arange(..., device="cpu"))

    But this results in an implicit host-device-copy and breaks cudagraphs.
    Rewrite the arange to use CUDA.
    """

@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(torch.ops.prims.convert_element_type.default, KeywordArg("arg"), KeywordArg("dtype1")),
        KeywordArg("dtype2"),
    ),
    pass_dict=patterns,
)
def pointless_convert(match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype):
    """Remove chain of dtype conversions often created by AMP"""

def definitely_equal(
    old_sizes: Sequence[torch.SymInt | int], new_sizes: Sequence[torch.SymInt | torch.fx.Node | int]
) -> bool:
    """
    Leverage guard_or_true/false to compare if two lists of int/symint are equal.
    Useful to compare sizes, strides etc.

    Can handle -1 in new_sizes which happens in the size arguments of a
    view op. old_sizes is supposed to be the tensor shape and should not
    contain -1.

    new_sizes can contains fx.Node when dynamic shape is enabled. In that
    case new_sizes[i].meta['val'] contains the real torch.SymInt.
    """

@register_graph_pattern(
    CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")), pass_dict=patterns
)
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""

@register_graph_pattern(
    CallFunction(
        aten.view.default, CallFunction(aten.view.default, KeywordArg("arg"), KeywordArg("size1")), KeywordArg("size2")
    ),
    pass_dict=patterns,
)
def pointless_view_pair(match: Match, arg, size1, size2):
    """Remove a pair of views that are pointless."""

@register_graph_pattern(
    CallFunction(
        aten.permute.default,
        CallFunction(aten.permute.default, KeywordArg("arg"), KeywordArg("perm1")),
        KeywordArg("perm2"),
    ),
    pass_dict=patterns,
)
def pointless_permute_pair(match: Match, arg, perm1, perm2): ...
@register_graph_pattern(CallFunction(aten.bmm, Arg(), Arg()), pass_dict=patterns)
def bmm_to_mm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    """Convert bmm to mm when batch size is 1"""

def mul_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=...): ...
def div_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=...): ...
