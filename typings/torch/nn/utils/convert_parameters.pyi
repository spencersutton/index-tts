from collections.abc import Iterable

import torch

def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """

def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    """
    Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
