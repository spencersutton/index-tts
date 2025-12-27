from dataclasses import dataclass

__all__ = ["Partial", "Placement", "Replicate", "Shard"]

class Placement:
    """
    The base class for the Placement type, where it describes how a DTensor is placed onto the
    ``DeviceMesh``. ``Placement`` and ``DeviceMesh`` together could describe the DTensor Layout.
    It is the base class of the three main DTensor Placement types: ``Shard``, ``Replicate``,
    and ``Partial``.

    This class is not meant to be used directly, mainly served as a typing stub.
    """
    def is_shard(self, dim: int | None = ...) -> bool: ...
    def is_replicate(self) -> bool: ...
    def is_partial(self, reduce_op: str | None = ...) -> bool: ...

@dataclass(frozen=True)
class Shard(Placement):
    """
    The ``Shard(dim)`` placement describes the DTensor sharding on tensor dimension
    ``dim`` over a corresponding ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension only holds a shard/piece of the global Tensor. The
    ``Shard(dim)`` placement follows the ``torch.chunk(dim)`` semantic, where the
    last few shards on the DeviceMesh dimension might be empty when the tensor dimension
    is not evenly divisible on the DeviceMesh dimension. The ``Shard`` placement can be
    used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    .. warning:: sharding on a tensor dimension where the tensor dimension size is not
        evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.
    """

    dim: int
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

kw_only_dataclass = ...

@dataclass(frozen=True, **kw_only_dataclass)
class _StridedShard(Shard):
    """
    _StridedShard is only introduced to support 2D FSDP2 + TP sharding where the tensor
    is sharded on the TP mesh dimension first, then sharded on the FSDP mesh dimension.
    We call this right-to-left sharding which is the opposite of the default
    left-to-right sharding. See the example below:
        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [Shard(0), Shard(0)]

    The default sharding behavior shards the tensor on "dp" mesh dimension first then
    "tp" dimension. The sharding result will be:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   1 (row 2-3)
        2       |   (1, 0)          |   2 (row 4-5)
        3       |   (1, 1)          |   3 (row 6-7)

    While the FSDP2 + TP sharding behavior does the opposite: it shards the tensor on
    "tp" mesh dim first then "dp" dim. This right-to-left sharding will produce the
    result:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The consequence is, any attempt to redistribute this DTensor to a full replica will
    produce a wrong result because the shard-to-replicate redistribution always happens
    right-to-left, regardless it's left-to-right sharding or right-to-left. To address
    this, we use _StridedShard placement to make this right-to-left sharding compatible
    with our left-to-right convention on both tensor distribution and redistribution.

    Now with _StridedShard, the right-to-left sharding above can be represented as:
        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [_StridedShard(0, split_factor=2), Shard(0)]

    And a left-to-right processing of `placements` will produce the same result, which is
    different from using the `Shard` placement:
        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The argument `split_factor` is the number of existing shards over the tensor sharding
    dimension before processing the _StridedShard placement, as if the sharding happened
    right-to-left. In the example above, the tensor should first be sharded on the "tp"
    dimension into 2 shards before being sharded on the "dp" dimension. Therefore, the
    `split_factor` of the _StridedShard placement on "dp" dim is 2.

    TODO: we should remove _StridedShard placement once we can unify it with Shard
    """

    split_factor: int
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass(frozen=True)
class Replicate(Placement):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. ``distribute_tensor``, ``DTensor.from_local``, etc.)
    """
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass(frozen=True)
class Partial(Placement):
    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
            to produce Replicated/Sharded DTensor. Only element-wise reduction operations
            are supported, including: "sum", "avg", "product", "max", "min", default: "sum".

    .. note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
        and can only be used by the ``DTensor.from_local`` API.
    """

    reduce_op: str = ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

_Partial = Partial
