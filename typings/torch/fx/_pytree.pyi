from typing import Any, Callable, Optional, TypeVar, TypeAlias
from torch.utils._pytree import PyTree, TreeSpec

FlattenFuncSpec: TypeAlias = Callable[[PyTree, TreeSpec], list]
FlattenFuncExactMatchSpec: TypeAlias = Callable[[PyTree, TreeSpec], bool]
SUPPORTED_NODES: dict[type[Any], FlattenFuncSpec] = ...
SUPPORTED_NODES_EXACT_MATCH: dict[type[Any], Optional[FlattenFuncExactMatchSpec]] = ...

def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = ...,
) -> None: ...
def tree_flatten_spec(pytree: PyTree, spec: TreeSpec) -> list[Any]: ...
