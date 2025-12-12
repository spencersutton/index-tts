from typing import Optional
from typing_extensions import TypeAlias
from torch import Tensor

TensorListList: TypeAlias = list[list[Optional[Tensor]]]
Indices: TypeAlias = list[int]
_foreach_supported_types = ...
