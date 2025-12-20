from kernels import Device, LayerRepository

_hub_kernels_available = ...
_KERNEL_MAPPING: dict[str, dict[Device | str, LayerRepository]] = ...

def is_hub_kernels_available():  # -> bool:
    ...

__all__ = [
    "LayerRepository",
    "is_hub_kernels_available",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
    "use_kernel_forward_from_hub",
]
