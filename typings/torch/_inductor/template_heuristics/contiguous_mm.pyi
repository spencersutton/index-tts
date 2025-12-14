from typing import TYPE_CHECKING

import torch

from ..kernel.mm import addmm_contiguous_subgraph_template, mm_contiguous_subgraph_template
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic

if TYPE_CHECKING: ...

@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(addmm_contiguous_subgraph_template.uid, None, op_name="addmm")
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics): ...

@register_template_heuristic(
    mm_contiguous_subgraph_template.uid, "cuda", register=torch.version.hip is not None, op_name="mm"
)
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, "cuda", register=torch.version.hip is not None, op_name="addmm"
)
class ContiguousMMHeuristics(GemmMaxAutotuneTemplateConfigHeuristics): ...
