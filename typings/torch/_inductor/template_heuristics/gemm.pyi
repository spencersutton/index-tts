from ..ir import Layout
from ..kernel_inputs import KernelInputs
from .base import TemplateConfigHeuristics

class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    def should_run(self, inputs: KernelInputs, layout: Layout) -> bool:
        """simple base override for GEMM family templates that run only in max-autotune"""
