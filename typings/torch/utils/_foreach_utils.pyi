from typing import Optional
from typing import TypeAlias
from torch import Tensor

type TensorListList = list[list[Tensor | None]]
type Indices = list[int]
_foreach_supported_types = ...
