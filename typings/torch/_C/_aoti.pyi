from ctypes import c_void_p
from typing import Protocol, overload

from torch import Tensor

def unsafe_alloc_void_ptrs_from_tensors(tensors: list[Tensor]) -> list[c_void_p]:
    """unsafe_alloc_void_ptrs_from_tensors(arg0: collections.abc.Sequence[torch.Tensor]) -> list[types.CapsuleType]"""

def unsafe_alloc_void_ptr_from_tensor(tensor: Tensor) -> c_void_p:
    """unsafe_alloc_void_ptr_from_tensor(arg0: torch.Tensor) -> types.CapsuleType"""

def alloc_tensors_by_stealing_from_void_ptrs(handles: list[c_void_p]) -> list[Tensor]:
    """alloc_tensors_by_stealing_from_void_ptrs(arg0: collections.abc.Sequence[types.CapsuleType]) -> list[torch.Tensor]"""

def alloc_tensor_by_stealing_from_void_ptr(handle: c_void_p) -> Tensor:
    """alloc_tensor_by_stealing_from_void_ptr(arg0: types.CapsuleType) -> torch.Tensor"""

class AOTIModelContainerRunner(Protocol):
    def run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self, tensor_map: dict[str, Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = ...
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerCpu:
    def __init__(self, model_so_path: str, num_models: int) -> None:
        """__init__(self: torch._C._aoti.AOTIModelContainerRunnerCpu, arg0: str, arg1: typing.SupportsInt) -> None"""
    def run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]:
        """run(self: torch._C._aoti.AOTIModelContainerRunnerCpu, inputs: collections.abc.Sequence[torch.Tensor], stream_handle: types.CapsuleType = None) -> list[torch.Tensor]"""
    def get_call_spec(self) -> list[str]:
        """get_call_spec(self: torch._C._aoti.AOTIModelContainerRunnerCpu) -> list[str]"""
    def get_constant_names_to_original_fqns(self) -> dict[str, str]:
        """get_constant_names_to_original_fqns(self: torch._C._aoti.AOTIModelContainerRunnerCpu) -> dict[str, str]"""
    def get_constant_names_to_dtypes(self) -> dict[str, int]:
        """get_constant_names_to_dtypes(self: torch._C._aoti.AOTIModelContainerRunnerCpu) -> dict[str, int]"""
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]:
        """extract_constants_map(self: torch._C._aoti.AOTIModelContainerRunnerCpu, arg0: bool) -> dict[str, torch.Tensor]"""
    def update_constant_buffer(
        self, tensor_map: dict[str, Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = ...
    ) -> None:
        """update_constant_buffer(self: torch._C._aoti.AOTIModelContainerRunnerCpu, tensor_map: collections.abc.Mapping[str, torch.Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = False) -> None"""
    def swap_constant_buffer(self) -> None:
        """swap_constant_buffer(self: torch._C._aoti.AOTIModelContainerRunnerCpu) -> None"""
    def free_inactive_constant_buffer(self) -> None:
        """free_inactive_constant_buffer(self: torch._C._aoti.AOTIModelContainerRunnerCpu) -> None"""

class AOTIModelContainerRunnerCuda:
    @overload
    def __init__(self, model_so_path: str, num_models: int) -> None: ...
    @overload
    def __init__(self, model_so_path: str, num_models: int, device_str: str) -> None: ...
    @overload
    def __init__(self, model_so_path: str, num_models: int, device_str: str, cubin_dir: str) -> None: ...
    def run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self, tensor_map: dict[str, Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = ...
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerXpu:
    @overload
    def __init__(self, model_so_path: str, num_models: int) -> None: ...
    @overload
    def __init__(self, model_so_path: str, num_models: int, device_str: str) -> None: ...
    @overload
    def __init__(self, model_so_path: str, num_models: int, device_str: str, kernel_bin_dir: str) -> None: ...
    def run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]: ...
    def get_call_spec(self) -> list[str]: ...
    def get_constant_names_to_original_fqns(self) -> dict[str, str]: ...
    def get_constant_names_to_dtypes(self) -> dict[str, int]: ...
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]: ...
    def update_constant_buffer(
        self, tensor_map: dict[str, Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = ...
    ) -> None: ...
    def swap_constant_buffer(self) -> None: ...
    def free_inactive_constant_buffer(self) -> None: ...

class AOTIModelContainerRunnerMps:
    def __init__(self, model_so_path: str, num_models: int) -> None:
        """__init__(self: torch._C._aoti.AOTIModelContainerRunnerMps, arg0: str, arg1: typing.SupportsInt) -> None"""
    def run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]:
        """run(self: torch._C._aoti.AOTIModelContainerRunnerMps, inputs: collections.abc.Sequence[torch.Tensor], stream_handle: types.CapsuleType = None) -> list[torch.Tensor]"""
    def get_call_spec(self) -> list[str]:
        """get_call_spec(self: torch._C._aoti.AOTIModelContainerRunnerMps) -> list[str]"""
    def get_constant_names_to_original_fqns(self) -> dict[str, str]:
        """get_constant_names_to_original_fqns(self: torch._C._aoti.AOTIModelContainerRunnerMps) -> dict[str, str]"""
    def get_constant_names_to_dtypes(self) -> dict[str, int]:
        """get_constant_names_to_dtypes(self: torch._C._aoti.AOTIModelContainerRunnerMps) -> dict[str, int]"""
    def extract_constants_map(self, use_inactive: bool) -> dict[str, Tensor]:
        """extract_constants_map(self: torch._C._aoti.AOTIModelContainerRunnerMps, arg0: bool) -> dict[str, torch.Tensor]"""
    def update_constant_buffer(
        self, tensor_map: dict[str, Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = ...
    ) -> None:
        """update_constant_buffer(self: torch._C._aoti.AOTIModelContainerRunnerMps, tensor_map: collections.abc.Mapping[str, torch.Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = False) -> None"""
    def swap_constant_buffer(self) -> None:
        """swap_constant_buffer(self: torch._C._aoti.AOTIModelContainerRunnerMps) -> None"""
    def free_inactive_constant_buffer(self) -> None:
        """free_inactive_constant_buffer(self: torch._C._aoti.AOTIModelContainerRunnerMps) -> None"""

class AOTIModelPackageLoader:
    def __init__(
        self, model_package_path: str, model_name: str, run_single_threaded: bool, num_runners: int, device_index: int
    ) -> None:
        """__init__(self: torch._C._aoti.AOTIModelPackageLoader, arg0: str, arg1: str, arg2: bool, arg3: typing.SupportsInt, arg4: typing.SupportsInt) -> None"""
    def get_metadata(self) -> dict[str, str]:
        """get_metadata(self: torch._C._aoti.AOTIModelPackageLoader) -> dict[str, str]"""
    def run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]:
        """run(self: torch._C._aoti.AOTIModelPackageLoader, inputs: collections.abc.Sequence[torch.Tensor], stream_handle: types.CapsuleType = None) -> list[torch.Tensor]"""
    def boxed_run(self, inputs: list[Tensor], stream_handle: c_void_p = ...) -> list[Tensor]:
        """boxed_run(self: torch._C._aoti.AOTIModelPackageLoader, inputs: list, stream_handle: types.CapsuleType = None) -> list"""
    def get_call_spec(self) -> list[str]:
        """get_call_spec(self: torch._C._aoti.AOTIModelPackageLoader) -> list[str]"""
    def get_constant_fqns(self) -> list[str]:
        """get_constant_fqns(self: torch._C._aoti.AOTIModelPackageLoader) -> list[str]"""
    def load_constants(
        self, constants_map: dict[str, Tensor], use_inactive: bool, check_full_update: bool, user_managed: bool = ...
    ) -> None:
        """load_constants(self: torch._C._aoti.AOTIModelPackageLoader, constants_map: collections.abc.Mapping[str, torch.Tensor], use_inactive: bool, check_full_update: bool, user_managed: bool = False) -> None"""
    def update_constant_buffer(
        self, tensor_map: dict[str, Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = ...
    ) -> None:
        """update_constant_buffer(self: torch._C._aoti.AOTIModelPackageLoader, tensor_map: collections.abc.Mapping[str, torch.Tensor], use_inactive: bool, validate_full_updates: bool, user_managed: bool = False) -> None"""
