import os

from torch._C._autograd import DeviceType, kineto_available
from torch._C._profiler import ProfilerActivity
from torch._environment import is_fbcode
from torch.autograd.profiler import record_function

from .profiler import (
    ExecutionTraceObserver,
    ProfilerAction,
    profile,
    schedule,
    supported_activities,
    tensorboard_trace_handler,
)

__all__ = [
    "DeviceType",
    "ExecutionTraceObserver",
    "ProfilerAction",
    "ProfilerActivity",
    "kineto_available",
    "profile",
    "record_function",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
]
if os.environ.get("KINETO_USE_DAEMON", "") or (is_fbcode() and os.environ.get("KINETO_FORCE_OPTIMIZER_HOOK", "")):
    _ = ...
