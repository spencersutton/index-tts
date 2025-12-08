from torch.distributed.tensor._api import (
    DTensor,
    distribute_module,
    distribute_tensor,
    empty,
    full,
    ones,
    rand,
    randn,
    zeros,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)

__all__ = [
    "DTensor",
    "Partial",
    "Placement",
    "Replicate",
    "Shard",
    "distribute_module",
    "distribute_tensor",
    "empty",
    "full",
    "ones",
    "rand",
    "randn",
    "zeros",
]
if DTensor not in _optim_foreach_supported_types: ...
if DTensor not in _util_foreach_supported_types: ...
