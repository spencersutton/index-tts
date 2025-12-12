from typing import Optional
from typing_extensions import override
from torch._inductor import ir
from .wrapper import PythonWrapperCodegen

class PythonWrapperMtia(PythonWrapperCodegen):
    @override
    def write_header(self) -> None: ...
    @override
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = ...,
    ) -> PythonWrapperCodegen: ...
