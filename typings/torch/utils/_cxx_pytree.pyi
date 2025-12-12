import functools
import sys
import types
import torch.utils._pytree as python_pytree
from collections.abc import Iterable
from typing import Any, Callable, Optional, TypeVar, Union, overload, TypeAlias, Self
from torch.utils._pytree import KeyEntry as KeyEntry
from optree import PyTreeSpec as TreeSpec

"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.
"""
if not python_pytree._cxx_pytree_dynamo_traceable: ...
__all__ = [
    "PyTree",
    "Context",
    "FlattenFunc",
    "UnflattenFunc",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "TreeSpec",
    "LeafSpec",
    "keystr",
    "key_get",
    "register_pytree_node",
    "tree_is_leaf",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_unflatten",
    "tree_iter",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_structure",
    "tree_map",
    "tree_map_with_path",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
]
__TORCH_DICT_SESSION = ...
T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")
Context: TypeAlias = Any
PyTree: TypeAlias = Any
FlattenFunc: TypeAlias = Callable[[PyTree], tuple[list[Any], Context]]
UnflattenFunc: TypeAlias = Callable[[Iterable[Any], Context], PyTree]
OpTreeUnflattenFunc: TypeAlias = Callable[[Context, Iterable[Any]], PyTree]
DumpableContext: TypeAlias = Any
ToDumpableContextFn: TypeAlias = Callable[[Context], DumpableContext]
FromDumpableContextFn: TypeAlias = Callable[[DumpableContext], Context]
KeyPath: TypeAlias = tuple[KeyEntry, ...]
FlattenWithKeysFunc: TypeAlias = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]

def register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    *,
    serialized_type_name: Optional[str] = ...,
    to_dumpable_context: Optional[ToDumpableContextFn] = ...,
    from_dumpable_context: Optional[FromDumpableContextFn] = ...,
    flatten_with_keys_fn: Optional[FlattenWithKeysFunc] = ...,
) -> None: ...
def tree_is_leaf(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> bool: ...
def tree_flatten(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> tuple[list[Any], TreeSpec]: ...
def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree: ...
def tree_iter(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> Iterable[Any]: ...
def tree_leaves(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> list[Any]: ...
def tree_structure(tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> TreeSpec: ...
def tree_map(
    func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...
def tree_map_(
    func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...

Type2: TypeAlias = tuple[type[T], type[S]]
Type3: TypeAlias = tuple[type[T], type[S], type[U]]
if sys.version_info >= (3, 10):
    TypeAny: TypeAlias = Union[type[Any], tuple[type[Any], ...], types.UnionType]
else: ...
Fn2: TypeAlias = Callable[[Union[T, S]], R]
Fn3: TypeAlias = Callable[[Union[T, S, U]], R]
Fn: TypeAlias = Callable[[T], R]
FnAny: TypeAlias = Callable[[Any], R]
MapOnlyFn: TypeAlias = Callable[[T], Callable[[Any], Any]]

@overload
def map_only(type_or_types_or_pred: type[T], /) -> MapOnlyFn[Fn[T, Any]]: ...
@overload
def map_only(type_or_types_or_pred: Type2[T, S], /) -> MapOnlyFn[Fn2[T, S, Any]]: ...
@overload
def map_only(type_or_types_or_pred: Type3[T, S, U], /) -> MapOnlyFn[Fn3[T, S, U, Any]]: ...
@overload
def map_only(type_or_types_or_pred: TypeAny, /) -> MapOnlyFn[FnAny[Any]]: ...
@overload
def map_only(type_or_types_or_pred: Callable[[Any], bool], /) -> MapOnlyFn[FnAny[Any]]: ...
def map_only(type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]], /) -> MapOnlyFn[FnAny[Any]]: ...
@overload
def tree_map_only(
    type_or_types_or_pred: type[T], /, func: Fn[T, Any], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: TypeAny, /, func: FnAny[Any], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...
@overload
def tree_map_only(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
def tree_map_only(
    type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: type[T], /, func: Fn[T, Any], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: Type2[T, S],
    /,
    func: Fn2[T, S, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: Type3[T, S, U],
    /,
    func: Fn3[T, S, U, Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: TypeAny, /, func: FnAny[Any], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...
@overload
def tree_map_only_(
    type_or_types_or_pred: Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
def tree_map_only_(
    type_or_types_or_pred: Union[TypeAny, Callable[[Any], bool]],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> PyTree: ...
def tree_all(pred: Callable[[Any], bool], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> bool: ...
def tree_any(pred: Callable[[Any], bool], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...) -> bool: ...
@overload
def tree_all_only(
    type_or_types: type[T], /, pred: Fn[T, bool], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> bool: ...
@overload
def tree_all_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> bool: ...
@overload
def tree_all_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> bool: ...
def tree_all_only(
    type_or_types: TypeAny, /, pred: FnAny[bool], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> bool: ...
@overload
def tree_any_only(
    type_or_types: type[T], /, pred: Fn[T, bool], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> bool: ...
@overload
def tree_any_only(
    type_or_types: Type2[T, S],
    /,
    pred: Fn2[T, S, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> bool: ...
@overload
def tree_any_only(
    type_or_types: Type3[T, S, U],
    /,
    pred: Fn3[T, S, U, bool],
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = ...,
) -> bool: ...
def tree_any_only(
    type_or_types: TypeAny, /, pred: FnAny[bool], tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> bool: ...
def broadcast_prefix(
    prefix_tree: PyTree, full_tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> list[Any]: ...
def treespec_dumps(treespec: TreeSpec, protocol: Optional[int] = ...) -> str: ...
@functools.lru_cache
def treespec_loads(serialized: str) -> TreeSpec: ...

class _DummyLeaf: ...

def treespec_pprint(treespec: TreeSpec) -> str: ...

class LeafSpecMeta(type(TreeSpec)):
    def __instancecheck__(self, instance: object) -> bool: ...

class LeafSpec(TreeSpec, metaclass=LeafSpecMeta):
    def __new__(cls) -> Self: ...

def tree_flatten_with_path(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> tuple[list[tuple[KeyPath, Any]], TreeSpec]: ...
def tree_leaves_with_path(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> list[tuple[KeyPath, Any]]: ...
def tree_map_with_path(
    func: Callable[..., Any], tree: PyTree, *rests: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = ...
) -> PyTree: ...
def keystr(kp: KeyPath) -> str: ...
def key_get(obj: Any, kp: KeyPath) -> Any: ...
