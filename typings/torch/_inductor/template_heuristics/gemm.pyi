from typing import TYPE_CHECKING

from ..ir import Layout
from ..kernel_inputs import KernelInputs
from .base import TemplateConfigHeuristics

if TYPE_CHECKING: ...

class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    def should_run(self, inputs: KernelInputs, layout: Layout) -> bool: ...
