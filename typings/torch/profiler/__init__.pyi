"""
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.
"""

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
