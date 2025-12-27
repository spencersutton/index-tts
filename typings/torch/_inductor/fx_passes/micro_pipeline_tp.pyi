from dataclasses import dataclass

import torch
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import Match

log = ...
aten = ...
patterns = ...

@dataclass
class _AllGatherMatch:
    """_AllGatherMatch(match: torch._inductor.pattern_matcher.Match, shard_node: torch.fx.node.Node, ag_node: torch.fx.node.Node, res_node: torch.fx.node.Node, gather_dim: int, group_name: str)"""

    match: Match
    shard_node: torch.fx.Node
    ag_node: torch.fx.Node
    res_node: torch.fx.Node
    gather_dim: int
    group_name: str
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def erase(self) -> None: ...

def find_all_gather_patterns(graph: torch.fx.Graph): ...

@dataclass
class _ReduceScatterMatch:
    """_ReduceScatterMatch(match: torch._inductor.pattern_matcher.Match, input_node: torch.fx.node.Node, reduce_scatter_node: torch.fx.node.Node, wait_tensor_node: torch.fx.node.Node, reduce_op: str, scatter_dim: int, group_name: str)"""

    match: Match
    input_node: torch.fx.Node
    reduce_scatter_node: torch.fx.Node
    wait_tensor_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_name: str
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def erase(self) -> None: ...

def find_reduce_scatter_patterns(graph: torch.fx.Graph): ...

@dataclass
class _Matmul:
    """_Matmul(nodes: list[torch.fx.node.Node], A_node: torch.fx.node.Node, B_node: torch.fx.node.Node, pre_mm_reshape: Optional[torch.fx.node.Node], post_mm_reshape: Optional[torch.fx.node.Node])"""

    nodes: list[torch.fx.Node]
    arg_ancestor_nodes: OrderedSet[torch.fx.Node] = ...
    A_node: torch.fx.Node
    B_node: torch.fx.Node
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None
    def __post_init__(self): ...
    def replace_with(self, new_node: torch.fx.Node) -> None:
        """Replace the matmul with the new node."""
    def erase(self) -> None: ...
    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> _Matmul: ...

@dataclass
class _ScaledMatmul(_Matmul):
    """_ScaledMatmul(nodes: list[torch.fx.node.Node], A_node: torch.fx.node.Node, B_node: torch.fx.node.Node, pre_mm_reshape: Optional[torch.fx.node.Node], post_mm_reshape: Optional[torch.fx.node.Node], A_scale_node: torch.fx.node.Node, B_scale_node: torch.fx.node.Node, bias_node: Optional[torch.fx.node.Node], result_scale_node: Optional[torch.fx.node.Node], out_dtype: Optional[torch.dtype], use_fast_accum: bool)"""

    A_scale_node: torch.fx.Node
    B_scale_node: torch.fx.Node
    bias_node: torch.fx.Node | None
    result_scale_node: torch.fx.Node | None
    out_dtype: torch.dtype | None
    use_fast_accum: bool
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None
    def __post_init__(self): ...
    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> _ScaledMatmul: ...

def fuse_all_gather_matmul(all_gather: _AllGatherMatch) -> None:
    """
    Fused the pattern

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    into

        A, Cs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """

def fuse_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None:
    """
    Fused the pattern

        reduce_scatter_tensor(A @ B, scatter_dim, group_name)

    into

        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )

    Returns boolean indicating if fusion was successful or not.
    """

def micro_pipeline_tp_pass(graph: torch.fx.Graph): ...
