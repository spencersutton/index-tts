from torch.nn.modules import Module

__all__ = ["orthogonal", "spectral_norm", "weight_norm"]

def orthogonal(
    module: Module, name: str = "weight", orthogonal_map: str | None = None, *, use_trivialization: bool = True
) -> Module: ...
def weight_norm[T: Module](module: T, name: str = "weight", dim: int = 0) -> T: ...
def spectral_norm(
    module: Module, name: str = "weight", n_power_iterations: int = 1, eps: float = 1e-12, dim: int | None = None
) -> Module: ...
