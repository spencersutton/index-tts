from enum import Enum

from torch import Tensor

def get_unwrapped(tensor: Tensor) -> Tensor:
    """get_unwrapped(arg0: torch.Tensor) -> torch.Tensor"""

def is_batchedtensor(tensor: Tensor) -> bool:
    """is_batchedtensor(arg0: torch.Tensor) -> bool"""

def is_functionaltensor(tensor: Tensor) -> bool:
    """is_functionaltensor(arg0: torch.Tensor) -> bool"""

def is_functorch_wrapped_tensor(tensor: Tensor) -> bool:
    """is_functorch_wrapped_tensor(arg0: torch.Tensor) -> bool"""

def is_gradtrackingtensor(tensor: Tensor) -> bool:
    """is_gradtrackingtensor(arg0: torch.Tensor) -> bool"""

def is_legacy_batchedtensor(tensor: Tensor) -> bool:
    """is_legacy_batchedtensor(arg0: torch.Tensor) -> bool"""

def maybe_get_bdim(tensor: Tensor) -> int:
    """maybe_get_bdim(arg0: torch.Tensor) -> int"""

def maybe_get_level(tensor: Tensor) -> int:
    """maybe_get_level(arg0: torch.Tensor) -> int"""

def maybe_current_level() -> int | None:
    """maybe_current_level() -> int | None"""

def unwrap_if_dead(tensor: Tensor) -> Tensor:
    """unwrap_if_dead(arg0: torch.Tensor) -> torch.Tensor"""

def current_level() -> int:
    """current_level() -> int"""

def count_jvp_interpreters() -> int: ...
def set_single_level_autograd_function_allowed(allowed: bool) -> None:
    """set_single_level_autograd_function_allowed(arg0: bool) -> None"""

def get_single_level_autograd_function_allowed() -> bool:
    """get_single_level_autograd_function_allowed() -> bool"""

class TransformType(Enum):
    """
    Members:

    Torch

    Grad

    Jvp

    Functionalize

    Vmap
    """

    Torch = ...
    Vmap = ...
    Grad = ...
    Jvp = ...
    Functionalize = ...

class RandomnessType(Enum):
    """
    Members:

    Error

    Same

    Different
    """

    Error = ...
    Same = ...
    Different = ...

class CInterpreter:
    def key(self) -> TransformType:
        """key(self: torch._C._functorch.CInterpreter) -> torch._C._functorch.TransformType"""
    def level(self) -> int:
        """level(self: torch._C._functorch.CInterpreter) -> int"""
    def serialize(self) -> bytes:
        """serialize(self: torch._C._functorch.CInterpreter) -> str"""
    @staticmethod
    def deserialize(bytes) -> CInterpreter:
        """deserialize(arg0: str) -> torch._C._functorch.CInterpreter"""

class CGradInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None:
        """__init__(self: torch._C._functorch.CGradInterpreterPtr, arg0: torch._C._functorch.CInterpreter) -> None"""
    def lift(self, Tensor) -> Tensor:
        """lift(self: torch._C._functorch.CGradInterpreterPtr, arg0: torch.Tensor) -> torch.Tensor"""
    def prevGradMode(self) -> bool:
        """prevGradMode(self: torch._C._functorch.CGradInterpreterPtr) -> bool"""

class CJvpInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None:
        """__init__(self: torch._C._functorch.CJvpInterpreterPtr, arg0: torch._C._functorch.CInterpreter) -> None"""
    def lift(self, Tensor) -> Tensor:
        """lift(self: torch._C._functorch.CJvpInterpreterPtr, arg0: torch.Tensor) -> torch.Tensor"""
    def prevFwdGradMode(self) -> bool:
        """prevFwdGradMode(self: torch._C._functorch.CJvpInterpreterPtr) -> bool"""

class CFunctionalizeInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None:
        """__init__(self: torch._C._functorch.CFunctionalizeInterpreterPtr, arg0: torch._C._functorch.CInterpreter) -> None"""
    def key(self) -> TransformType:
        """key(self: torch._C._functorch.CFunctionalizeInterpreterPtr) -> torch._C._functorch.TransformType"""
    def level(self) -> int:
        """level(self: torch._C._functorch.CFunctionalizeInterpreterPtr) -> int"""
    def functionalizeAddBackViews(self) -> bool:
        """functionalizeAddBackViews(self: torch._C._functorch.CFunctionalizeInterpreterPtr) -> bool"""

class CVmapInterpreterPtr:
    def __init__(self, interpreter: CInterpreter) -> None:
        """__init__(self: torch._C._functorch.CVmapInterpreterPtr, arg0: torch._C._functorch.CInterpreter) -> None"""
    def key(self) -> TransformType:
        """key(self: torch._C._functorch.CVmapInterpreterPtr) -> torch._C._functorch.TransformType"""
    def level(self) -> int:
        """level(self: torch._C._functorch.CVmapInterpreterPtr) -> int"""
    def batchSize(self) -> int:
        """batchSize(self: torch._C._functorch.CVmapInterpreterPtr) -> Union[int, torch.SymInt]"""
    def randomness(self) -> RandomnessType:
        """randomness(self: torch._C._functorch.CVmapInterpreterPtr) -> torch._C._functorch.RandomnessType"""

class DynamicLayer: ...

def get_dynamic_layer_stack_depth() -> int:
    """get_dynamic_layer_stack_depth() -> int"""

def get_interpreter_stack() -> list[CInterpreter]:
    """get_interpreter_stack() -> list[at::functorch::Interpreter] | None"""

def peek_interpreter_stack() -> CInterpreter:
    """peek_interpreter_stack() -> at::functorch::Interpreter | None"""

def pop_dynamic_layer_stack() -> DynamicLayer:
    """pop_dynamic_layer_stack() -> at::functorch::DynamicLayer"""

def pop_dynamic_layer_stack_and_undo_to_depth(int) -> None:
    """pop_dynamic_layer_stack_and_undo_to_depth(arg0: typing.SupportsInt) -> None"""

def push_dynamic_layer_stack(dl: DynamicLayer) -> int:
    """push_dynamic_layer_stack(arg0: at::functorch::DynamicLayer) -> int"""
