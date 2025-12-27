from collections.abc import Callable

from torch import fx, nn

from .aot_autograd import make_boxed_compiler

log = ...

@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """

def draw_graph_compile(name): ...
@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.
    """

class DebugInterpreter(fx.Interpreter):
    def run(self, *args): ...
    def run_node(self, n): ...

@make_boxed_compiler
def debug_nop(fx_g: fx.GraphModule, _) -> Callable:
    """
    Returns a (slow) interpreter over the FX graph module that also checks
    various debugging properties (e.g., that tracing strides matched real
    strides.)
    """

@make_boxed_compiler
def simple_ts_compile(fx_g, _): ...
def nnc_jit(f): ...

aten = ...
default_decompositions = ...
default_decompositions = ...

@make_boxed_compiler
def print_compile(fx_g, _): ...
def memory_efficient_fusion(fn: Callable | nn.Module, **kwargs):
    """
    Wrapper function over :func:`aot_function` and :func:`aot_module` to perform
    memory efficient fusion. It uses the
    :func:`min_cut_rematerialization_partition` partitioner to perform efficient
    recomputation. It uses NVFuser to compile the generated forward and backward
    graphs.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Union[Callable, nn.Module]): A Python function or a ``nn.Module``
            that takes one or more arguments. Must return one or more Tensors.
        **kwargs: Any other overrides you want to make to the settings

    Returns:
        Returns a ``Callable``  or ``nn.Module`` that retains the eager behavior
        of the original :attr:`fn`, but whose forward and backward graphs have
        gone through recomputation optimizations, and the graphs have been
        compiled with nvfuser.
    """

def debug_compile(fx_g, inps): ...

graph_index = ...

def get_inputs(input_data_path):
    """Return a random input for the given inputs meta generated from _save_fx_default."""

def graph_dumper_aot(current_name, folder_name, dump_example_input=...):
    """
    Dump the forward, backward, and joint computation graph.
    Example Usage:
    save_fx_func = graph_dumper_aot(current_name, folder_name, dump_example_input = False)
    optimize_ctx = torchdynamo.optimize(
        save_fx_func
    )
    with torch.enable_grad():
        with optimize_ctx:
            result = forward_and_backward_pass(model, example_inputs)
    """
