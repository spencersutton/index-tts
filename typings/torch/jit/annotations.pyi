import enum

import torch

if torch.distributed.rpc.is_available(): ...

class Module:
    def __init__(self, name, members) -> None: ...
    def __getattr__(self, name): ...

class EvalEnv:
    env = ...
    def __init__(self, rcb) -> None: ...
    def __getitem__(
        self, name
    ) -> (
        type[Tensor | Tuple[Any, ...] | List[Any] | Dict[Any, Any] | Optional | Union | Future[Any] | _Await[Any]]
        | Module
        | Any
        | None
    ): ...

def get_signature(fn, rcb, loc, is_method) -> tuple[list[Any], Any] | None: ...
def is_function_or_method(the_callable) -> TypeIs[FunctionType] | TypeIs[MethodType]: ...
def is_vararg(the_callable) -> bool: ...
def get_param_names(fn, n_args) -> list[str]: ...
def check_fn(fn, loc) -> None: ...
def parse_type_line(type_line, rcb, loc) -> tuple[list[Any], Any]:
    """
    Parse a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """

def get_type_line(source) -> None:
    """Try to find the line containing a comment with the type annotation."""

def split_type_line(type_line) -> tuple[Any, Any]:
    """
    Split the comment with the type annotation into parts for argument and return types.

    For example, for an input of:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor, Tensor]

    This function will return:
        ("(Tensor, torch.Tensor)", "Tuple[Tensor, Tensor]")
    """

def try_real_annotations(fn, loc) -> tuple[list[Any], Any] | None:
    """Try to use the Py3.5+ annotation syntax to get the type."""

def get_enum_value_type(e: type[enum.Enum], loc) -> AnyType | JitType: ...
def is_tensor(ann) -> bool: ...
def try_ann_to_type(ann, loc, rcb=...): ...
def ann_to_type(ann, loc, rcb=...): ...

__all__ = [
    "Any",
    "AnyType",
    "BroadcastingList1",
    "BroadcastingList2",
    "BroadcastingList3",
    "ComplexType",
    "Dict",
    "DictType",
    "FloatType",
    "IntType",
    "List",
    "ListType",
    "Module",
    "StringType",
    "TensorType",
    "Tuple",
    "TupleType",
    "ann_to_type",
    "check_fn",
    "get_param_names",
    "get_signature",
    "get_type_line",
    "is_dict",
    "is_list",
    "is_optional",
    "is_tuple",
    "is_union",
    "parse_type_line",
    "split_type_line",
    "try_ann_to_type",
    "try_real_annotations",
]
