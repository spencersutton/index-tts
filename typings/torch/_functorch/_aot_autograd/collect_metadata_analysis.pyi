from collections.abc import Callable

from torch import Tensor

from .descriptors import AOTInput
from .schemas import ViewAndMutationMeta

zip = ...
log = ...
static_input_logger = ...

def coerce_tangent_and_suggest_memory_format(x: Tensor): ...
def run_functionalized_fw_and_collect_metadata(
    f,
    *,
    flat_args_descs: list[AOTInput],
    keep_input_mutations: bool,
    is_train: bool = ...,
    static_input_indices: list[int] | None = ...,
    pre_dispatch: bool = ...,
    is_export: bool = ...,
) -> Callable[..., ViewAndMutationMeta]: ...
