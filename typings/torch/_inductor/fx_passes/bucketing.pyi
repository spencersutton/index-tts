import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._ordered_set import OrderedSet

logger: logging.Logger = ...

def bucket_cap_mb_by_bucket_idx_default(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """

def bucket_all_gather(
    gm: torch.fx.GraphModule, bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = ..., mode: str | None = ...
) -> None: ...
def bucket_reduce_scatter(
    gm: torch.fx.GraphModule, bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = ..., mode: str | None = ...
) -> None: ...
def is_all_gather_into_tensor(node: torch.fx.Node) -> bool: ...
def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool: ...
def is_wait_tensor(node: torch.fx.Node) -> bool: ...
def is_wait_tensor_from_all_gather_into_tensor(node: torch.fx.Node) -> bool: ...
def collect_node_descendants(graph: torch.fx.Graph) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
    """
    Collects the descendants of each node in the graph.
    Args:
        graph (torch.fx.Graph): The graph to collect descendants from.
    Returns:
        dict[torch.fx.Node, OrderedSet[torch.fx.Node]]: A dictionary mapping each node to its descendants.
    """

def greedy_bucket_collective_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[torch.fx.Node], bool],
    node_group_key: Callable[[torch.fx.Node], Any],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = ...,
) -> list[list[torch.fx.Node]]:
    """
    Bucketing adjacent collectives with equal node_group_key.
    We can not bucket non adjacent collectives,
    as this will effectively change the order of collectives.
    Reordering can lead to different order on different ranks.
    """

def bucket_all_gather_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = ...,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all all_gather nodes and groups them into buckets,
    based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (torch.fx.GraphModule): GraphModule where to bucket all_gathers.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets at the start,
            as first all_gather is usually exposed.  Interface of bucket_cap_mb_by_bucket_idx
            is `bucket_cap_mb_by_bucket_idx_default` function that is default value for `bucket_cap_mb_by_bucket_idx`.
        filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
            only all_gather nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of all_gather nodes.
    """

def bucket_reduce_scatter_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = ...,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all reduce_scatter nodes and groups them into buckets,
        based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (torch.fx.GraphModule): GraphModule where to bucket reduce_scatters.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets.
        filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
            only reduce_scatter nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of reduce_scatter nodes.
    """

def reduce_scatter_merge_fn_to_trace_custom_ops(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]: ...
def reduce_scatter_merge_fn_to_trace(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]: ...
def all_gather_merge_fn_to_trace_custom_ops(
    ag_ins: list[torch.Tensor], group_size: int, group_name: str, dtype: torch.dtype, rank: int
) -> list[torch.Tensor]: ...
def all_gather_merge_fn_to_trace(
    ag_ins: list[torch.Tensor], group_size: int, group_name: str, dtype: torch.dtype, rank: int
) -> list[torch.Tensor]: ...
def all_gather_merge_fn_to_trace_functional(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,
    rank: int,
    use_fsdp_ag_copy_in: bool = ...,
) -> list[torch.Tensor]: ...
def merge_reduce_scatter(
    gm: torch.fx.GraphModule, rs_buckets: list[list[torch.fx.Node]], mode: str | None = ...
) -> None:
    """Merges specified buckets of reduce_scatter to joint reduce_scatter."""

def merge_all_gather(gm: torch.fx.GraphModule, ag_buckets: list[list[torch.fx.Node]], mode: str | None = ...) -> None:
    """Merges specified buckets of all_gather to joint all_gather."""
