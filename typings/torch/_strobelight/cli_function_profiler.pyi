from typing import Any, Callable, Optional, TypeVar
from typing_extensions import ParamSpec

logger = ...
console_handler = ...
formatter = ...
_P = ParamSpec("_P")
_R = TypeVar("_R")

class StrobelightCLIProfilerError(Exception): ...

class StrobelightCLIFunctionProfiler:
    _lock = ...
    def __init__(
        self,
        *,
        stop_at_error: bool = ...,
        max_profile_duration_sec: int = ...,
        sample_each: float = ...,
        run_user_name: str = ...,
        timeout_wait_for_running_sec: int = ...,
        timeout_wait_for_finished_sec: int = ...,
        recorded_env_variables: Optional[list[str]] = ...,
        sample_tags: Optional[list[str]] = ...,
        stack_max_len: int = ...,
        async_stack_max_len: int = ...,
    ) -> None: ...
    def profile(self, work_function: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> Optional[_R]: ...

def strobelight(
    profiler: Optional[StrobelightCLIFunctionProfiler] = ..., **kwargs: Any
) -> Callable[[Callable[_P, _R]], Callable[_P, Optional[_R]]]: ...
