from collections.abc import Callable

from .triton_compat import triton

log = ...

def get_field(config, name): ...
def set_field(config, name, value): ...

class CoordescTuner:
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    TODO will it be necessary to tune multiple fields simultaneously.


    TODO: what if both increasing and decreasing a field can improve perf.
          i.e., there are multiple local optima..
    """
    def __init__(self, is_mm=..., name=..., size_hints=..., inductor_meta=...) -> None: ...
    def get_config_max(self, prefix: str) -> int: ...
    def get_warpsmax(self): ...
    def cache_benchmark_result(self, config, timing): ...
    def lookup_in_cache(self, config): ...
    def call_func(self, func, config): ...
    @property
    def tunable_fields(self): ...
    def value_too_large(self, name: str, val: int) -> bool: ...
    def get_neighbour_values(self, name, orig_val, radius=..., include_self=...):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as it's own neighbour.
        """
    @staticmethod
    def has_improvement(baseline, test): ...
    def check_all_tuning_directions(self, func: Callable[[triton.Config], float], best_config, best_timing):
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
    def compare_config(self, func, candidate_config, best_config, best_timing):
        """
        Check if candidate_config is better than best_config.

        Return a tuple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        """
    def autotune(
        self,
        func: Callable[[triton.Config], float],
        baseline_config: triton.Config,
        baseline_timing: float | None = ...,
    ) -> triton.Config: ...
