from typing import Any

from ..ir import Layout
from ..kernel_inputs import KernelInputs
from .base import TemplateConfigHeuristics

class AddMMConfigMixin(TemplateConfigHeuristics):
    def get_extra_kwargs(self, kernel_inputs: KernelInputs, layout: Layout, op_name: str) -> dict[str, Any]: ...
