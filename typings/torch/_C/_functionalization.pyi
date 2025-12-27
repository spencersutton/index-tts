"""functionalization related pybind."""

from torch import Tensor
from torch.types import _bool

class ViewMeta:
    has_symbolic_inputs: _bool

def get_view_meta_sequence(tensor: Tensor) -> list[ViewMeta]:
    """get_view_meta_sequence(arg0: torch.Tensor) -> list[at::functionalization::ViewMeta]"""

def apply_view_meta_sequence(base: Tensor, sequence: list[ViewMeta]) -> Tensor:
    """apply_view_meta_sequence(arg0: torch.Tensor, arg1: collections.abc.Sequence[at::functionalization::ViewMeta]) -> torch.Tensor"""
