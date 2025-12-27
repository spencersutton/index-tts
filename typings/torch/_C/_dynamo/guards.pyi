"""Module containing checks on tensors"""

import enum
from collections.abc import Callable
from typing import Any

import torch

type GuardManagerType = enum.Enum

class GlobalStateGuard:
    """Guard on PyTorch global flags such as no_grad"""
    def check(self) -> bool:
        """Return true if global state was the same as at creation time"""
    def reason(self) -> str:
        """Return string reason for guard check failing"""

class LeafGuard:
    def verbose_code_parts(self) -> list[str]:
        """verbose_code_parts(self: torch._C._dynamo.guards.LeafGuard) -> list"""

class RelationalGuard: ...

class GuardDebugInfo:
    verbose_code_parts: list[str]
    result: bool
    num_guards_executed: int

class GuardManager:
    def check(self, value: Any) -> bool: ...
    def check_verbose(self, value: Any) -> GuardDebugInfo: ...
    def globals_dict_manager(
        self, f_globals: dict[str, Any], source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """globals_dict_manager(self: torch._C._dynamo.guards.GuardManager, f_globals: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def framelocals_manager(
        self, key: tuple[str, int], source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """framelocals_manager(self: torch._C._dynamo.guards.GuardManager, key: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def dict_getitem_manager(
        self, key: Any, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """dict_getitem_manager(self: torch._C._dynamo.guards.GuardManager, key: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def grad_manager(self, source: str, example_value: Any, guard_manager_enum: GuardManagerType) -> GuardManager:
        """grad_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def generic_getattr_manager(
        self, attr: str, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """generic_getattr_manager(self: torch._C._dynamo.guards.GuardManager, attr: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def getitem_manager(
        self, key: Any, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """getitem_manager(self: torch._C._dynamo.guards.GuardManager, key: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def get_generic_dict_manager(
        self, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """get_generic_dict_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def list_getitem_manager(
        self, key: Any, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """list_getitem_manager(self: torch._C._dynamo.guards.GuardManager, key: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def tuple_getitem_manager(
        self, key: Any, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """tuple_getitem_manager(self: torch._C._dynamo.guards.GuardManager, key: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def set_getitem_manager(
        self, index: Any, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """set_getitem_manager(self: torch._C._dynamo.guards.GuardManager, index: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def func_defaults_manager(
        self, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """func_defaults_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def func_kwdefaults_manager(
        self, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """func_kwdefaults_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def tuple_iterator_getitem_manager(
        self, index: Any, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """tuple_iterator_getitem_manager(self: torch._C._dynamo.guards.GuardManager, index: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def weakref_call_manager(
        self, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """weakref_call_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def call_function_no_args_manager(
        self, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """call_function_no_args_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def global_weakref_manager(
        self, global_name: str, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """global_weakref_manager(self: torch._C._dynamo.guards.GuardManager, global_name: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def type_manager(self, source: str, example_value: Any, guard_manager_enum: GuardManagerType) -> GuardManager:
        """type_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def getattr_manager(
        self, attr: str, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """getattr_manager(self: torch._C._dynamo.guards.GuardManager, attr: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def tensor_property_size_manager(
        self, idx: int, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """tensor_property_size_manager(self: torch._C._dynamo.guards.GuardManager, idx: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def tensor_property_shape_manager(
        self, idx: int, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager: ...
    def tensor_property_storage_offset_manager(
        self, idx: int, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """tensor_property_storage_offset_manager(self: torch._C._dynamo.guards.GuardManager, idx: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def indexed_manager(
        self, idx: int, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """indexed_manager(self: torch._C._dynamo.guards.GuardManager, idx: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def lambda_manager(
        self, python_lambda: Callable[..., Any], source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """lambda_manager(self: torch._C._dynamo.guards.GuardManager, python_lambda: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def get_root(self) -> RootGuardManager: ...
    def get_source(self) -> str:
        """get_source(self: torch._C._dynamo.guards.GuardManager) -> str"""
    def fail_count(self) -> int:
        """fail_count(self: torch._C._dynamo.guards.GuardManager) -> int"""
    def get_child_managers(self) -> list[GuardManager]:
        """get_child_managers(self: torch._C._dynamo.guards.GuardManager) -> list[torch._C._dynamo.guards.GuardManager]"""
    def repr(self) -> str: ...
    def type_of_guarded_value(self) -> str: ...
    def get_leaf_guards(self) -> list[LeafGuard]:
        """get_leaf_guards(self: torch._C._dynamo.guards.GuardManager) -> list[torch._C._dynamo.guards.LeafGuard]"""
    def get_accessors(self) -> list[GuardManager]:
        """get_accessors(self: torch._C._dynamo.guards.GuardManager) -> list[torch._C._dynamo.guards.GuardAccessor]"""
    def is_guarded_value_immutable(self) -> bool:
        """is_guarded_value_immutable(self: torch._C._dynamo.guards.GuardManager) -> bool"""
    def is_tag_safe(self) -> bool:
        """is_tag_safe(self: torch._C._dynamo.guards.GuardManager) -> bool"""
    def is_tag_safe_root(self) -> bool:
        """is_tag_safe_root(self: torch._C._dynamo.guards.GuardManager) -> bool"""
    def has_no_accessors(self) -> bool:
        """has_no_accessors(self: torch._C._dynamo.guards.GuardManager) -> bool"""
    def has_object_aliasing_guard(self) -> bool:
        """has_object_aliasing_guard(self: torch._C._dynamo.guards.GuardManager) -> bool"""
    def get_type_of_guarded_value(self) -> type:
        """get_type_of_guarded_value(self: torch._C._dynamo.guards.GuardManager) -> object"""
    def type_dict_manager(self, source: str, example_value: Any, guard_manager_enum: GuardManagerType) -> GuardManager:
        """type_dict_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def type_mro_manager(self, source: str, example_value: Any, guard_manager_enum: GuardManagerType) -> GuardManager:
        """type_mro_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def code_manager(self, source: str, example_value: Any, guard_manager_enum: GuardManagerType) -> GuardManager:
        """code_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def closure_manager(self, source: str, example_value: Any, guard_manager_enum: GuardManagerType) -> GuardManager:
        """closure_manager(self: torch._C._dynamo.guards.GuardManager, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def add_lambda_guard(self, user_lambda: Callable[..., Any], verbose_code_parts: list[str]) -> None:
        """add_lambda_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_id_match_guard(self, id_val: int, verbose_code_parts: list[str]) -> None:
        """add_id_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_equals_match_guard(self, equals_val: Any, verbose_code_parts: list[str]) -> None:
        """add_equals_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_global_state_guard(self, initial_state: Any, verbose_code_parts: list[str]) -> None:
        """add_global_state_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_torch_function_mode_stack_guard(self, initial_stack: list[Any], verbose_code_parts: list[str]) -> None:
        """add_torch_function_mode_stack_guard(self: torch._C._dynamo.guards.GuardManager, arg0: list, arg1: object) -> None"""
    def add_mapping_keys_guard(self, value: Any, verbose_code_parts: list[str]) -> None:
        """add_mapping_keys_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_dict_length_check_guard(self, value: int, verbose_code_parts: list[str]) -> None:
        """add_dict_length_check_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_length_check_guard(self, value: int, verbose_code_parts: list[str]) -> None:
        """add_length_check_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_true_match_guard(self, verbose_code_parts: list[str]) -> None:
        """add_true_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object) -> None"""
    def add_false_match_guard(self, verbose_code_parts: list[str]) -> None:
        """add_false_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object) -> None"""
    def add_none_match_guard(self, verbose_code_parts: list[str]) -> None:
        """add_none_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object) -> None"""
    def add_not_none_guard(self, verbose_code_parts: list[str]) -> None:
        """add_not_none_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object) -> None"""
    def add_dispatch_key_set_guard(self, dispatch_key: Any, verbose_code_parts: list[str]) -> None:
        """add_dispatch_key_set_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_tensor_match_guard(
        self,
        value: Any,
        sizes: list[int],
        strides: list[int],
        tensor_name: str,
        verbose_code_parts: list[str],
        ptype: Any,
        dispatch_keys: Any,
    ) -> None:
        """add_tensor_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object, arg2: object, arg3: object, arg4: object, arg5: object, arg6: object) -> None"""
    def add_dynamic_indices_guard(self, value: set[Any], verbose_code_parts: list[str]) -> None:
        """add_dynamic_indices_guard(self: torch._C._dynamo.guards.GuardManager, arg0: set, arg1: object) -> None"""
    def add_no_hasattr_guard(self, attr_name: str, verbose_code_parts: list[str]) -> None:
        """add_no_hasattr_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_dict_contains_guard(self, contains: bool, key: Any, verbose_code_parts: list[str]) -> None:
        """add_dict_contains_guard(self: torch._C._dynamo.guards.GuardManager, arg0: bool, arg1: object, arg2: object) -> None"""
    def add_type_match_guard(self, value: int, verbose_code_parts: list[str]) -> None:
        """add_type_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_dict_version_guard(self, value: Any, verbose_code_parts: list[str]) -> None:
        """add_dict_version_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object) -> None"""
    def add_set_contains_guard(self, contains: bool, item: Any, verbose_code_parts: list[str]) -> None:
        """add_set_contains_guard(self: torch._C._dynamo.guards.GuardManager, arg0: bool, arg1: object, arg2: object) -> None"""
    def add_tuple_iterator_length_guard(self, length: int, type_id: int, verbose_code_parts: list[str]) -> None:
        """add_tuple_iterator_length_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object, arg2: object) -> None"""
    def add_range_iterator_match_guard(
        self, start: int, stop: int, step: int, type_id: int, verbose_code_parts: list[str]
    ) -> None:
        """add_range_iterator_match_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object, arg1: object, arg2: object, arg3: object, arg4: object) -> None"""
    def add_default_device_guard(self, verbose_code_parts: list[str]) -> None:
        """add_default_device_guard(self: torch._C._dynamo.guards.GuardManager, arg0: object) -> None"""
    def mark_tag_safe(self) -> None:
        """mark_tag_safe(self: torch._C._dynamo.guards.GuardManager) -> None"""
    def mark_tag_safe_root(self) -> None:
        """mark_tag_safe_root(self: torch._C._dynamo.guards.GuardManager) -> None"""

class RootGuardManager(GuardManager):
    def get_epilogue_lambda_guards(self) -> list[LeafGuard]:
        """get_epilogue_lambda_guards(self: torch._C._dynamo.guards.RootGuardManager) -> list[torch._C._dynamo.guards.LeafGuard]"""
    def add_epilogue_lambda_guard(self, guard: LeafGuard, verbose_code_parts: list[str]) -> None:
        """add_epilogue_lambda_guard(self: torch._C._dynamo.guards.RootGuardManager, arg0: object, arg1: object) -> None"""
    def clone_manager(self, clone_filter_fn: Callable[[GuardManager], bool]) -> RootGuardManager:
        """clone_manager(self: torch._C._dynamo.guards.RootGuardManager, arg0: collections.abc.Callable) -> torch._C._dynamo.guards.RootGuardManager"""
    def attach_compile_id(self, compile_id: str) -> None:
        """attach_compile_id(self: torch._C._dynamo.guards.RootGuardManager, arg0: str) -> None"""

class DictGuardManager(GuardManager):
    def get_key_manager(
        self, index: int, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """get_key_manager(self: torch._C._dynamo.guards.DictGuardManager, index: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def get_value_manager(
        self, index: int, source: str, example_value: Any, guard_manager_enum: GuardManagerType
    ) -> GuardManager:
        """get_value_manager(self: torch._C._dynamo.guards.DictGuardManager, index: object, source: str, example_value: object, guard_manager_enum: object) -> torch._C._dynamo.guards.GuardManager"""
    def get_key_value_managers(self) -> dict[int, tuple[GuardManager, GuardManager]]:
        """get_key_value_managers(self: torch._C._dynamo.guards.DictGuardManager) -> dict[int, tuple[torch._C._dynamo.guards.GuardManager, torch._C._dynamo.guards.GuardManager]]"""

class GuardAccessor: ...
class DictGetItemGuardAccessor(GuardAccessor): ...
class GetGenericDictGuardAccessor(GuardAccessor): ...
class TypeDictGuardAccessor(GuardAccessor): ...
class TypeMROGuardAccessor(GuardAccessor): ...
class ClosureGuardAccessor(GuardAccessor): ...
class TupleGetItemGuardAccessor(GuardAccessor): ...
class TypeGuardAccessor(GuardAccessor): ...
class CodeGuardAccessor(GuardAccessor): ...
class FuncDefaultsGuardAccessor(GuardAccessor): ...
class FuncKwDefaultsGuardAccessor(GuardAccessor): ...

class GetAttrGuardAccessor(GuardAccessor):
    def get_attr_name(self) -> str:
        """get_attr_name(self: torch._C._dynamo.guards.GetAttrGuardAccessor) -> str"""

def install_object_aliasing_guard(x: GuardManager, y: GuardManager, verbose_code_parts: list[str]) -> None:
    """install_object_aliasing_guard(arg0: torch._C._dynamo.guards.GuardManager, arg1: torch._C._dynamo.guards.GuardManager, arg2: object) -> None"""

def install_no_tensor_aliasing_guard(
    guard_managers: list[GuardManager], tensor_names: list[str], verbose_code_parts: list[str]
) -> None:
    """install_no_tensor_aliasing_guard(arg0: list, arg1: list, arg2: object) -> None"""

def install_storage_overlapping_guard(
    overlapping_guard_managers: list[GuardManager],
    non_overlapping_guard_managers: list[GuardManager],
    verbose_code_parts: list[str],
) -> None:
    """install_storage_overlapping_guard(arg0: list, arg1: list, arg2: object) -> None"""

def install_symbolic_shape_guard(
    guard_managers: list[GuardManager],
    nargs_int: int,
    nargs_float: int,
    py_addr: int,
    py_addr_keep_alive: Any,
    verbose_code_parts: list[str],
) -> None:
    """install_symbolic_shape_guard(arg0: list, arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: typing.SupportsInt, arg4: object, arg5: object) -> None"""

def profile_guard_manager(guard_manager: GuardManager, f_locals: dict[str, Any], n_iters: int) -> float:
    """profile_guard_manager(arg0: torch._C._dynamo.guards.RootGuardManager, arg1: object, arg2: typing.SupportsInt) -> float"""

class TensorGuards:
    """Check properties of a torch.Tensor"""
    def __init__(
        self,
        *,
        dynamic_dims_sizes: list[torch.SymInt | None] | None = ...,
        dynamic_dims_strides: list[torch.SymInt | None] | None = ...,
    ) -> None: ...
    def check(self, *args: Any) -> bool: ...
    def check_verbose(self, *args: Any, tensor_check_names: list[str] | None = ...) -> bool | str:
        """verbose fail reasons for failed checks"""

def assert_size_stride(
    item: torch.Tensor, size: torch.types._size, stride: torch.types._size, op_name: str | None = ...
) -> None: ...
def assert_alignment(item: torch.Tensor, alignment: int, op_name: str | None = ...) -> None: ...
def check_obj_id(obj: object, expected: int) -> bool: ...
def check_type_id(obj: object, expected: int) -> bool: ...
def dict_version(d: dict[Any, Any]) -> int: ...
def compute_overlapping_tensors(tensors: list[torch.Tensor], symbolic: bool = ...) -> set[int]:
    """compute_overlapping_tensors(tensors: collections.abc.Sequence[torch.Tensor], symbolic: bool = True) -> set[int]"""
