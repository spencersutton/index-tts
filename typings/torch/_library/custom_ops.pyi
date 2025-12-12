import weakref
import torch
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Literal, Optional, Union, overload, TypeAlias
from torch import _C
from torch.types import _dtype
from torch.utils._exposed_in import exposed_in

device_types_t: TypeAlias = Optional[Union[str, Sequence[str]]]
log = ...

@overload
def custom_op(
    name: str,
    fn: None = ...,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = ...,
    schema: Optional[str] = ...,
) -> Callable[[Callable[..., object]], CustomOpDef]: ...
@overload
def custom_op(
    name: str,
    fn: Callable[..., object],
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = ...,
    schema: Optional[str] = ...,
) -> CustomOpDef: ...
@exposed_in("torch.library")
def custom_op(
    name: str,
    fn: Optional[Callable] = ...,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = ...,
    schema: Optional[str] = ...,
    tags: Optional[Sequence[_C.Tag]] = ...,
) -> Union[Callable[[Callable[..., object]], CustomOpDef], CustomOpDef]: ...

class CustomOpDef:
    def __init__(
        self, namespace: str, name: str, schema: str, fn: Callable, tags: Optional[Sequence[_C.Tag]] = ...
    ) -> None: ...
    @contextmanager
    def set_kernel_enabled(self, device_type: str, enabled: bool = ...):  # -> Generator[None, Any, None]:

        ...
    def register_kernel(self, device_types: device_types_t, fn: Optional[Callable] = ..., /) -> Callable: ...
    def register_fake(self, fn: Callable, /) -> Callable: ...
    def register_torch_dispatch(self, torch_dispatch_class: Any, fn: Optional[Callable] = ..., /) -> Callable: ...
    def register_autograd(self, backward: Callable, /, *, setup_context: Optional[Callable] = ...) -> None: ...
    def __call__(self, *args, **kwargs):  # -> Any:
        ...
    def register_vmap(self, func: Optional[Callable] = ...):  # -> Callable[..., None] | None:

        ...
    def register_autocast(self, device_type: str, cast_inputs: _dtype):  # -> Callable[..., Any]:

        ...

def increment_version(val: Any) -> None: ...

OPDEF_TO_LIB: dict[str, torch.library.Library] = ...
OPDEFS: weakref.WeakValueDictionary = ...

def get_library_allowing_overwrite(namespace: str, name: str) -> torch.library.Library: ...
