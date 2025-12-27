import logging
from collections.abc import Callable

import torch

logger: logging.Logger = ...

def is_graph_input(node: torch.fx.Node) -> bool: ...
def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool: ...
def is_graph_output(node: torch.fx.Node) -> bool: ...
def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool: ...
def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule, bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = ..., mode: str | None = ...
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Optional[Callable[[int], float]]): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """

def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule, bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = ..., mode: str | None = ...
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Optional[Callable[[int], float]]): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.
    """
