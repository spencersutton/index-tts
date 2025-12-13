import torch
from typing import Optional, TYPE_CHECKING
from torch.export.graph_signature import ExportGraphSignature

if TYPE_CHECKING: ...

def replace_autocast_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]: ...
