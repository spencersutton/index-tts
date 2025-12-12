import contextlib
import logging
import queue
import warnings
import torch.fx
import types
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
from typing_extensions import Self, final, override
from torch._inductor.metrics import CachedMetricsDeltas
from torch._inductor.output_code import CompiledFxGraphConstants, CompiledFxGraphConstantsWithGm, OutputCode
from torch._subclasses import FakeTensorMode
from torch.utils._ordered_set import OrderedSet
from . import config
from .compile_fx import FxCompile, _CompileFxKwargs
from collections.abc import Generator, Sequence
from torch._inductor.utils import InputType
from torch.fx import GraphModule

if TYPE_CHECKING: ...

@dataclass
class _VirtualizedSerializer:
    aot_compilation: Any = ...
    choices: Any = ...
    local_buffer_context: Any = ...
    ops: Any = ...
    kernel: Any = ...
    current_node: Any = ...
    @classmethod
    def serialize(cls) -> _VirtualizedSerializer: ...
    def patch(self) -> _VirtualizedSerializerContextManager: ...

class _VirtualizedSerializerContextManager(contextlib.ExitStack):
    def __init__(self, virtualized: _VirtualizedSerializer) -> None: ...
    @override
    def __enter__(self) -> Self: ...

class _LoweringSerializer:
    fallbacks: OrderedSet[str]
    def __init__(self) -> None: ...
    def patch(self) -> _LoweringSerializerContextManager: ...

class _LoweringSerializerContextManager(contextlib.ExitStack):
    def __init__(self, lowering: _LoweringSerializer) -> None: ...
    @override
    def __enter__(self) -> Self: ...

@dataclass
class _FakeTensorModeSerializer:
    allow_non_fake_inputs: bool
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...
    @contextlib.contextmanager
    def patch(self, fake_mode: FakeTensorMode) -> Generator[None, None, None]: ...

@dataclass
class _WireProtocolInput:
    gm: torch.fx.GraphModule
    example_inputs: Sequence[InputType]
    inputs_to_check: Sequence[int]
    graph_kwargs: _CompileFxKwargs
    tracing_context: Optional[torch._guards.TracingContext]
    config: dict[str, object]
    virtualized: _VirtualizedSerializer
    deterministic_guard_for_testing: Optional[torch.testing._internal.common_utils.DeterministicGuard]
    logger_state: _LoggerState
    lowering: _LoweringSerializer
    fake_tensor_mode: _FakeTensorModeSerializer
    def serialize(self) -> _WireProtocolPickledInput: ...

@dataclass
class _WireProtocolPickledInput:
    value: bytes
    def deserialize(self) -> _WireProtocolInput: ...

@dataclass
class _WireProtocolOutput:
    graph: OutputCode
    metrics: CachedMetricsDeltas
    logs: list[logging.LogRecord]
    warning_replay: Optional[list[warnings.WarningMessage]]
    shape_env: Optional[torch.fx.experimental.symbolic_shapes.ShapeEnv]
    def serialize(self) -> _WireProtocolPickledOutput: ...

@dataclass
class _WireProtocolPickledOutput:
    value: bytes
    def deserialize(self, constants: CompiledFxGraphConstants) -> _WireProtocolOutput: ...

class _LoggerState:
    loggers: dict[str, int]
    captured_logs: Optional[_CapturedLogs] = ...
    def __init__(self) -> None: ...
    def __enter__(self) -> _CapturedLogs: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None: ...

class _CapturedLogs:
    state: _LoggerState
    queue: queue.Queue[logging.LogRecord]
    handlers: Optional[dict[str, logging.Handler]]
    def __init__(self, state: _LoggerState) -> None: ...
    def finish(self) -> list[logging.LogRecord]: ...
    def remove(self) -> None: ...
    def apply(self) -> None: ...

class _SerializedFxCompile(FxCompile):
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...
    def serialize_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> Optional[tuple[_WireProtocolPickledInput, CompiledFxGraphConstantsWithGm]]: ...

@final
class _DebugSerdeFxCompile(_SerializedFxCompile): ...

class _OutOfProcessFxCompile(_SerializedFxCompile): ...

@final
class _DebugFileFxCompile(_SerializedFxCompile):
    file_index = ...
