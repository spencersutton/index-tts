import torch
from typing import Optional, TYPE_CHECKING
from torch.export.exported_program import ModuleCallSignature
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.export.graph_signature import ExportGraphSignature

if TYPE_CHECKING: ...
__all__ = ["CollectTracepointsPass"]

class CollectTracepointsPass(PassBase):
    def __init__(self, specs: dict[str, ModuleCallSignature], sig: ExportGraphSignature) -> None: ...
    def call(self, gm: torch.fx.GraphModule) -> Optional[PassResult]: ...
