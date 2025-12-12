from typing import Any, TYPE_CHECKING
from .base import TemplateConfigHeuristics
from ..ir import Layout
from ..kernel_inputs import KernelInputs

if TYPE_CHECKING: ...

class AddMMConfigMixin(TemplateConfigHeuristics):
    def get_extra_kwargs(self, kernel_inputs: KernelInputs, layout: Layout, op_name: str) -> dict[str, Any]: ...
