import dataclasses
from typing import Any, Self

from torch import dtype as torch_dtype

from ..ir import GraphPartitionSignature
from ..utils import IndentedBuffer, cache_on_self
from .cpp_wrapper_cpu import CppWrapperCpu
from .wrapper import PythonWrapperCodegen

_cpp_string_literal_escapes = ...
_cpp_string_literal_pattern = ...

def cpp_string_literal(s: str) -> str: ...

@dataclasses.dataclass
class DeferredTritonCallWrapper:
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]
    def generate(self, wrapper: CppWrapperGpu):
        """Generate the GPU kernel definition, as well as load and launch code."""
    def generate_grid(self, prefix: IndentedBuffer, inductor_meta: dict[str, Any], params: dict[str, Any]): ...
    def generate_load_kernel(self, prefix, kernel_var_name, params): ...
    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params): ...

class CppWrapperGpu(CppWrapperCpu):
    """Generates cpp wrapper for running on GPU and calls CUDA kernels"""
    def __init__(self) -> None: ...
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = ...,
    ): ...
    def write_header(self): ...
    @cache_on_self
    def write_tma_descriptor_helpers_once(self): ...
    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str: ...
    def get_autotuning_input_name(self, idx): ...
    def codegen_inputs(self): ...
    def generate(self, is_inference): ...
    def finalize_prefix(self):
        """Define the triton kernels now that autotuning is finished"""
    def generate_tma_descriptor(self, desc): ...
    def generate_args_decl(
        self,
        code: IndentedBuffer | Self,
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel=...,
        scratch_spaces: dict[str, int] | None = ...,
    ):
        """
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        """
    @staticmethod
    def prepare_triton_wrapper_args(call_args: list[Any], arg_types: list[Any]) -> tuple[list[Any], list[Any]]: ...
    def make_zero_buffer(self, name): ...

@dataclasses.dataclass
class UnwrapUnspecArg:
    """Marker that we need to call .item() on the tensor"""

    dtype: torch_dtype
