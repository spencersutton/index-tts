import dataclasses
import sys
import sympy
from typing import Callable, Literal, Optional, TYPE_CHECKING, TypeVar, Union, overload, TypeAlias
from torch.utils._ordered_set import OrderedSet
from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode

T = TypeVar("T")
U = TypeVar("U")
Split: TypeAlias = tuple[sympy.Expr, ...]
VarsAndRanges: TypeAlias = tuple[list[sympy.Symbol], list[sympy.Expr]]
loop_tiling_log = ...
if TYPE_CHECKING: ...

def solve_for_zero(expr: sympy.Expr) -> Optional[sympy.Expr]: ...
def solve_for_tiling(expr: sympy.Expr) -> Optional[sympy.Expr]: ...
def find_coalesced_var(index: sympy.Expr, var_ranges: dict[sympy.Expr, int]) -> Optional[sympy.Expr]: ...

@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: dict[sympy.Expr, OrderedSet[str]]
    writes: dict[sympy.Expr, OrderedSet[str]]
    var_ranges: dict[sympy.Symbol, int]

@overload
def get_pw_red_splits(
    n: SchedulerNode, pointwise_numel: sympy.Expr, red_numel: sympy.Expr, none_if_not_divisible: Literal[True]
) -> Optional[tuple[VarsAndRanges, VarsAndRanges]]: ...
@overload
def get_pw_red_splits(
    n: SchedulerNode, pointwise_numel: sympy.Expr, red_numel: sympy.Expr, none_if_not_divisible: Literal[False] = ...
) -> tuple[VarsAndRanges, VarsAndRanges]: ...
def get_pw_red_splits(
    n: SchedulerNode, pointwise_numel: sympy.Expr, red_numel: sympy.Expr, none_if_not_divisible: bool = ...
) -> Optional[tuple[VarsAndRanges, VarsAndRanges]]: ...

class NodeSplitGetter:
    def __init__(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None: ...
    def get_node_splits(self) -> tuple[Split, Split]: ...
    def try_split(self, pw: Split, red: Split) -> Optional[tuple[Split, Split]]: ...

if sys.version_info >= (3, 10):
    zip_equal = ...
else: ...

def apply_var_mapping(
    iter_vars: list[sympy.Symbol],
    red_vars: list[sympy.Symbol],
    norm_pw_vars: list[sympy.Symbol],
    norm_red_vars: list[sympy.Symbol],
    new_ranges: list[list[sympy.Expr]],
    return_getters_groups: list[list[Callable[[list[sympy.Expr]], sympy.Expr]]],
) -> dict[sympy.Symbol, sympy.Expr]: ...
def extract_normalized_read_writes(
    node: Union[FusedSchedulerNode, SchedulerNode],
) -> Optional[FusedNormalizedReadsWrites]: ...
def get_score(addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int]) -> int: ...
def get_hint(v: Union[sympy.Expr, int]) -> int: ...

@dataclasses.dataclass(frozen=True)
class VarTiling:
    var: sympy.Symbol
    tiling_factor: int
    score: int

@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    coalesced_by_var: dict[sympy.Expr, int]
    norm_read_writes: FusedNormalizedReadsWrites
    suggested_split: Optional[VarTiling] = ...

def analyze_memory_coalescing(
    fused_node: Union[FusedSchedulerNode, SchedulerNode],
) -> Optional[CoalesceVarAnalysis]: ...
