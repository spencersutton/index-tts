from typing import TYPE_CHECKING
from .base import TemplateConfigHeuristics
from ..ir import Layout
from ..kernel_inputs import KernelInputs

if TYPE_CHECKING: ...

class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    def should_run(self, inputs: KernelInputs, layout: Layout) -> bool: ...
