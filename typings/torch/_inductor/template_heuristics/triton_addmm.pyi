from typing import TYPE_CHECKING, Any

from ..ir import Layout
from ..kernel_inputs import KernelInputs
from .base import TemplateConfigHeuristics

if TYPE_CHECKING: ...

class AddMMConfigMixin(TemplateConfigHeuristics):
    def get_extra_kwargs(self, kernel_inputs: KernelInputs, layout: Layout, op_name: str) -> dict[str, Any]: ...
