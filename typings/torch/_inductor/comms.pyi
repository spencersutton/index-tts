from dataclasses import dataclass

import torch
from torch._inductor.scheduler import BaseSchedulerNode

from .ir import IRNode, Operation

log = ...
overlap_log = ...

def align_runtime_estimations_across_all_distributed_ranks(snodes: list[BaseSchedulerNode]): ...
def sink_waits(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Greedily schedules waits as late as possible."""

def raise_comms(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Greedily schedules comms as early as possible."""

def reorder_compute_for_overlap(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    This achieves the following overall scheduling procedure:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """

def reorder_communication_preserving_peak_memory(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Reorders communication ops relative to computation ops to improve communication-compute overlapping and hide comm
    latency.  Stops moving a particular op if it reaches a point that would have increased the peak memory footprint.

    Currently, follows these heuristics (subject to change or tune):
    - never reorders collectives relative to one another, for SPMD safety
    - has an option for per-collective prefetch limit, but does not enable it by default
    - limits the total number of reorder steps to some factor of the graph size to prevent worst-case quadratic
      performance

    Prerequisite: sink_comms_and_waits - ensure comm and wait nodes are scheduled as late as possible, respecting data
    dependencies.  That allows reorder_communication_preserving_peak_memory to take a best case peak-memory snapshot,
    and then monotonically improve latency by moving collectives backward in time.

    Peak memory impact is computed in an iterative fashion.  First, memory use at each timestep is computed, and global
    peak memory is computed as a max over timesteps.  Then, when swapping any two adjacent nodes, only the curr-memory
    for the earlier of the nodes after the swap is affected.  This enables checking step by step whether a swap is
    peak-memory-safe, and bailing out if not.  Example:

    0   n0      C0
    1   n1      C0 + Allocs(n1) - Frees(n1)
    2   n2      C0 + Allocs(n1) - Frees(n1) + Allocs(n2) - Frees(n2)

    0   n0      C0
    1   n2      C0 + Allocs(n2) - Frees(n2)    <-- After moving n2 to Time 1, only time1 memory changes
    2   n1      C0 + Allocs(n2) - Frees(n2) + Allocs(n1) - Frees(n1)
    """

@dataclass
class ReorderInfo:
    """Debug info describing how an individual snode was reordered"""

    initial_exposed: float = ...
    final_exposed: float = ...
    limiting_factor: str = ...
    moves: int = ...
    grouped: int = ...
    grouped_info: str = ...
    @property
    def improvement(self): ...

def is_gemm_like(node: IRNode | Operation | None) -> bool: ...
def contains_gemm_like(snode: BaseSchedulerNode) -> bool: ...
def decide_global_ordering_of_comms(
    nodes: list[BaseSchedulerNode], name_to_buf, name_to_fused_node
) -> list[BaseSchedulerNode]:
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """

@dataclass
class SinkWaitInfo:
    """SinkWaitInfo(grouped: 'int' = 0, grouped_info: 'str' = '', moves: 'int' = 0, moves_info: 'str' = '', limiting_factor: 'str' = 'None')"""

    grouped: int = ...
    grouped_info: str = ...
    moves: int = ...
    moves_info: str = ...
    limiting_factor: str = ...

def sink_waits_iterative(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]: ...
def estimate_op_runtime(snode: BaseSchedulerNode) -> float:
    """Returns estimated op runtime in nanoseconds (ns)"""

def node_summary(snode): ...
def visualize_overlap(order): ...
def reorder_compute_and_comm_for_overlap(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]: ...
def remove_fsdp2_unsharded_param_graph_input_usage(graph: torch.fx.Graph):
    """
    This FX graph pass replaces uses of FSDP2 unsharded params with their corresponding
    graph intermediates that were fsdp.copy_ into the unsharded params in the original graph.

    NOTE: Can only apply this pass to any of the FSDP2 unsharded params that have this pattern
    (or repetition of): `resize_(full) -> copy_ -> resize_(0)`. Because of this, for partial-graph case
    where `resize_(full) -> copy_` is in one graph and `resize_(0)` is in another graph, we can't
    remove these resize and copy ops and thus we will have worse performance there.

    In other words, "do we try to remove all the resize_(full) -> copy_ -> resize_(0) nodes for this unsharded param"
    is actually a per-unsharded-param decision, since for each unsharded param, we look at its resize sequence pattern
    (in `check_resize_pattern()`) to determine if its set of resize and copy nodes can be removed.
    """

def reinplace_fsdp_all_gather(graph: torch.fx.Graph) -> None: ...
def get_op_idx(snode): ...
def enforce_comm_ordering_for_fsdp(
    snodes: list[torch._inductor.scheduler.BaseSchedulerNode],
    name_to_buf: dict[str, torch._inductor.scheduler.SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
) -> list[torch._inductor.scheduler.BaseSchedulerNode]: ...
