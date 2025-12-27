from ..ir import GraphPartitionSignature
from .cpp_wrapper_gpu import CppWrapperGpu
from .wrapper import PythonWrapperCodegen

class CppWrapperMps(CppWrapperGpu):
    """Generates cpp wrapper for running on MPS and calls metal kernels"""
    def __init__(self) -> None: ...
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = ...,
    ) -> CppWrapperMps: ...
    def write_mps_kernel_call(self, name: str, call_args: list[str]) -> None: ...
    @staticmethod
    def get_device_include_path(device: str) -> str: ...
    def codegen_additional_funcs(self) -> None:
        """
        We want to codegen the mps kernel function variable initializations
        ahead of time.  This is so that if we reuse kernels within subgraphs, we
        don't need to worry about the scope in which we're initializing the
        variables. Instead we will just initialize the variables all at the top
        level.

        The kernel function variable initializations should look something like:
        ```
        const std::shared_ptr<at::native::mps::MetalKernelFunction> get_mps_lib_0() {
            static const auto func = mps_lib_0.getKernelFunction("generated_kernel");
            return func;
        }
        AOTIMetalKernelFunctionHandle get_mps_lib_0_handle() {
            static const auto handle = AOTIMetalKernelFunctionHandle(get_mps_lib_0().get());
            return handle;
        }
        ```
        """
