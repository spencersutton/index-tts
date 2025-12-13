import time
from types import CodeType, FrameType
from typing import Any


class ExecutionInfo:
    def __init__(self, code_obj: CodeType) -> None:
        self.code_obj = code_obj
        self.start_time = 0.0
        self.cumulative_time = 0.0
        self.call_count = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.call_count += 1

    def stop(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.cumulative_time += elapsed


dict_calls: dict[CodeType, ExecutionInfo] = {}


def profile_func(frame: FrameType, event: str, _arg: Any) -> None:
    assert event in ("call", "return", "c_call", "c_return", "c_exception")

    match event:
        case "call":
            if frame.f_code not in dict_calls:
                dict_calls[frame.f_code] = ExecutionInfo(frame.f_code)

            dict_calls[frame.f_code].start()
        case "return":
            if frame.f_code in dict_calls:
                dict_calls[frame.f_code].stop()
        case _:
            ...
