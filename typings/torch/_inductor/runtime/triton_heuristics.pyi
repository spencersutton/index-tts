import dataclasses
from collections.abc import Callable, Container
from typing import Any, Literal, TypeVar

from torch._guards import CompileId
from torch.utils._ordered_set import OrderedSet

from .autotune_cache import AutotuneCache
from .hints import AutotuneHint, DeviceProperties, HeuristicType
from .static_cuda_launcher import StaticallyLaunchedCudaKernel
from .triton_compat import CompiledKernel, Config, KernelInterface

class InductorConfig(Config):
    """Inductor-specific Triton config with additional control flags"""
    def __init__(self, *args, dynamic_scale_rblock=..., **kwargs) -> None: ...

class InductorConfig(Config):
    """Inductor-specific Triton config with additional control flags"""
    def __init__(self, *args, dynamic_scale_rblock=..., **kwargs) -> None: ...

class NoTritonConfigsError(RuntimeError): ...

type LauncherType = Any
type _KernelType = CompiledKernel | StaticallyLaunchedCudaKernel
_T = TypeVar("_T", bound=_KernelType)
log = ...
triton_name_sub = ...

def generate_lookup_hash_from_source_code(size_hints_str: str, source_code: str) -> str: ...
def lookup_autotune_config(size_hints, fn) -> Config | None: ...
def get_total_reduction_numel(numels: dict[str, int]) -> int: ...
def autotune_hints_to_configs(
    hints: OrderedSet[AutotuneHint], size_hints, block_size: int, device_props: DeviceProperties
) -> list[Config]:
    """
    AutotuneHints can be attached to the metadata of triton kernels for providing
    suggestions about what to try for autotuning. One reason to do this is if there are
    some configs that are only useful in specific scenarios, in which case we can avoid
    wasting compile time on autotuning unless we know we are in one of those scenarios.

    Based on those hints, this function will generate a list of additional autotuning
    configs to try.
    """

def disable_pointwise_autotuning(inductor_meta): ...
def check_autotune_cache(
    configs: list[Config], filename: str | None, inductor_meta: dict[str, Any]
) -> tuple[list[Config], AutotuneCache | None, dict[str, Any]]:
    """Given a list of configs, checks autotune cache and return metadata"""

class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """
    def __init__(
        self,
        fn,
        triton_meta,
        configs,
        save_cache_hook,
        mutated_arg_names: list[str],
        optimize_mem,
        heuristic_type,
        size_hints=...,
        inductor_meta=...,
        custom_kernel=...,
        filename: str | None = ...,
        reset_to_zero_arg_names: list[str] | None = ...,
        autotune_cache_info: dict[str, Any] | None = ...,
    ) -> None: ...
    def is_statically_launchable(self):
        """
        Checks if every compiled kernel is statically launchable, which
        allows us to efficiently cache it in FXGraphCache
        """
    def recheck_autotune_cache(self, reload_kernel_from_src: Callable[[], CachingAutotuner]) -> None:
        """
        On cache load on static autotuner, we need to recheck the autotune cache, since
        a best config could have been found from a previous run
        """
    def set_compile_info(self, compile_id: CompileId | None, is_backward: bool) -> None: ...
    def precompile(
        self,
        warm_cache_only=...,
        reload_kernel: Callable[[], CachingAutotuner] | None = ...,
        static_triton_bundle_key: str | None = ...,
    ): ...
    def prepare_for_pickle(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        """
        Drop stuff from triton.JITFunction that does not pickle.
        This must be called after precompile so that these things are no longer needed.
        Returns a tuple of old values
        """
    def restore_after_unpickle(self, old_values: tuple[Any, Any, Any, Any, Any, Any] | None) -> None: ...
    def prepare_for_caching(self) -> None:
        """
        Statically Launched CUDA Kernels have a raw cubin on them
        that we don't need to store in the cache(since TritonBundler handles the collection for us)
        """
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def get_device_interface(self): ...
    def bench(self, launcher, *args, with_profiler=..., **kwargs):
        """Measure the performance of a given launcher"""
    def copy_args_to_cpu_if_needed(self, *args, **kwargs):
        """
        To support benchmarking in the presence of mutated args, we need to avoid
        autotuning contanminating them. We try to pass cloned args to the kernel.
        If those clones would increase the peak memory usage, however, we instead
        copy to cpu and restore them after each iteration. Figure out the args
        to be copied and do the copying.
        """
    def restore_args_from_cpu(self, cpu_copies): ...
    def reset_to_zero_args(self, *args, **kwargs): ...
    def maybe_clone_args(self, exclude: Container[str], *args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
        """
        Prepare new args and kwargs by cloning any in-place buffers
        (that are not in the provided exclusion list), to avoid autotune
        contaminating them. Avoid cloning the other buffers because it
        leads to increased memory usage.
        """
    def clone_args(self, *args, **kwargs) -> tuple[list[Any], dict[str, Any]]: ...
    def benchmark_all_configs(self, *args, **kwargs): ...
    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
    def save_gpu_kernel(self, stream, launcher): ...
    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate desecnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
    def get_profiler_kwargs(self, stream, launcher): ...
    def run(self, *args, stream, benchmark_run=..., **kwargs): ...

class _ConstRepr:
    def __init__(self, value: str) -> None: ...
    def __call__(self, _=...) -> str: ...

class CompileResult[T: _KernelType]:
    """Base class representing compiled result."""
    def __init__(
        self, kernel: _T, config: Config, compile_meta: dict[str, Any], inductor_meta: dict[str, Any]
    ) -> None: ...
    def make_launcher(self) -> LauncherType: ...

class CannotStaticallyLaunchKernel(Exception): ...

class StaticTritonCompileResult(CompileResult[StaticallyLaunchedCudaKernel]):
    """
    TritonCompileResult that uses StaticCudaLauncher,
    which vastly simplifies the setup and metadata needed to be kept.
    """
    @staticmethod
    def can_statically_launch(
        kernel: CompiledKernel,
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
        heuristic_type: HeuristicType,
    ) -> StaticallyLaunchedCudaKernel | None: ...
    def reload_cubin_path(self):
        """
        When loading from cache on disk, we want to reload cubin
        files from their appropriate location on disc.
        """
    def make_launcher(self) -> LauncherType: ...

class TritonCompileResult(CompileResult[CompiledKernel]):
    """
    Upstream Triton CompileKernel can not be pickled.  This is a wrapper
    to support serialization and generate the launcher function.
    """
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def make_launcher(self) -> LauncherType:
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """

collected_calls: list[Any] = ...

def start_graph(): ...
def end_graph(output_file): ...

class DebugAutotuner(CachingAutotuner):
    def __init__(self, *args, regex_filter=..., with_profiler=..., with_bandwidth_info=..., **kwargs) -> None: ...
    def run(self, *args, stream, **kwargs): ...

def hash_configs(configs: list[Config]):
    """Hash used to check for changes in configurations"""

def cached_autotune(
    size_hints: list[int] | None,
    configs: list[Config],
    triton_meta,
    heuristic_type,
    filename=...,
    inductor_meta=...,
    custom_kernel=...,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """

def unique_configs(configs: list[Config]):
    """Remove duplicate configurations"""

def check_config(cfg, *, xnumel=..., ynumel=..., znumel=...): ...
def check_max_block(cfg: dict[str, int]):
    """Check that block sizes are within the maximum allowed."""

def triton_config(
    size_hints, x, y=..., z=..., num_stages=..., num_elements_per_warp=..., min_elem_per_thread=...
) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.

    num_elements_per_warp is a suggestion for controlling how many warps
    the triton config should contain. e.g.: if x=16, y=8, z=4 then
    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
    just a suggestion, and sometimes other adjustment heuristics will
    override the num_elements_per_warp.

    min_elem_per_thread controls the minimum number of elements
    processed by each thread. It's always enforced.
    """

def triton_config_reduction(
    size_hints, x: int, r: int, num_stages=..., num_warps=..., register_intensive=..., dynamic_scale_rblock=...
) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """

def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=..., register_intensive=...):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """

def pointwise(size_hints, triton_meta, tile_hint=..., filename=..., min_elem_per_thread=..., inductor_meta=...):
    """Construct @triton.heuristics() based on size_hints."""

def match_target_block_product(size_hints, tiling_scores, target_block_product, min_block_size=...):
    """
    Distribute block sizes across dimensions according to tiling scores,
    aiming to match a target product of block sizes.
    """

def adapt_config_for_tiling(
    size_hints,
    tiling_scores,
    original_x,
    original_r,
    num_warps=...,
    num_stages=...,
    register_intensive=...,
    persistent_reduction=...,
) -> Config:
    """
    Create an adapted configuration based on tiling scores,
    redistributing the same total block size (x * r) according to tiling scores.
    """

def reduction(size_hints, reduction_hint=..., triton_meta=..., filename=..., inductor_meta=...):
    """args to @triton.heuristics()"""

def cooperative_reduction(size_hints, reduction_hint, triton_meta, filename, inductor_meta): ...
def persistent_reduction(size_hints, reduction_hint=..., triton_meta=..., filename=..., inductor_meta=...): ...
def split_scan(size_hints, reduction_hint=..., triton_meta=..., filename=..., inductor_meta=...):
    """Heuristic for TritonSplitScanKernel"""

def template(
    num_stages,
    num_warps,
    triton_meta,
    num_consumer_groups=...,
    num_buffers_warp_spec=...,
    filename=...,
    inductor_meta=...,
):
    """Compile a triton template"""

def config_to_dict(config: Config) -> dict[str, Any]: ...
def config_from_dict(config: dict[str, Any]) -> Config: ...
def fixed_config(config, filename, triton_meta, inductor_meta):
    """Used when the configuration is already decided at compile time"""

def user_autotune(configs, triton_meta, filename=..., inductor_meta=..., custom_kernel=...):
    """Compile a user defined triton kernel"""

def foreach(triton_meta, num_warps, filename=..., inductor_meta=...):
    """Compile a triton foreach kernel"""

@dataclasses.dataclass
class GridExpr:
    """Generate code for grid size expressions in launcher"""

    inductor_meta: dict[str, Any]
    mode: Literal["python", "cpp", "python_slow"] = ...
    prefix: list[str] = ...
    x_grid: str | int = ...
    y_grid: str | int = ...
    z_grid: str | int = ...
    def __post_init__(self) -> None: ...
    def generate(self, meta: dict[str, int]) -> None: ...
    def ceildiv(self, numel: str | int, block: None | int | str) -> str | int: ...
    def maximum(self, seq: list[int | str]) -> int | str:
        """Codegen for max function with constant folding, constants are represented as int"""
    def summation(self, seq: list[int | str]) -> int | str:
        """Codegen for sum function with constant folding, constants are represented as int"""
    def assign_tmp(self, name: str, expr: str | int) -> str: ...
    @staticmethod
    def from_meta(
        inductor_meta: dict[str, Any], cfg: Config | dict[str, int], mode: Literal["python", "cpp", "python_slow"] = ...
    ) -> GridExpr: ...
    def eval_slow(self, meta: dict[str, int]) -> tuple[int, int, int]: ...

class Grid1D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class Grid2D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class Grid3D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class Grid2DWithYZOverflow(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class CooperativeReductionGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class SplitScanGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class FixedGrid(GridExpr):
    @staticmethod
    def setup_grid_as_args() -> dict[str, Any]:
        """Inductor meta so the launcher takes three extra grid arguments"""
    def generate(self, meta: dict[str, int]) -> None: ...

class PrecomputedGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None: ...

class ComboKernelGrid(GridExpr):
    def generate(self, meta: dict[str, int]): ...
    def combo_x_grid(self, xnumels: list[int | str], no_x_dims: list[bool], meta: dict[str, int]) -> str | int: ...

class SequentialComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(self, xnumels: list[int | str], no_x_dims: list[bool], meta: dict[str, int]) -> str | int: ...

class RoundRobinComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(self, xnumels: list[int | str], no_x_dims: list[bool], meta: dict[str, int]) -> str: ...
