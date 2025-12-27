"""distributed autograd bindings"""

import torch

class DistAutogradContext: ...

def backward(context_id: int, roots: list[torch.Tensor], retain_graph: bool = ...) -> None:
    """
    backward(context_id: int, roots: List[Tensor], retain_graph = False) -> None

    Kicks off the distributed backward pass using the provided roots. This
    currently implements the :ref:`fast-mode-algorithm` which
    assumes all RPC messages sent in the same distributed autograd context
    across workers would be part of the autograd graph during the backward pass.

    We use the provided roots to discover the autograd graph and compute
    appropriate dependencies. This method blocks until the entire
    autograd computation is done.

    We accumulate the gradients in the appropriate
    :class:`torch.distributed.autograd.context` on each of the nodes. The autograd
    context to be used is looked up given the ``context_id`` that is passed in when
    :meth:`torch.distributed.autograd.backward` is called. If there is no valid
    autograd context corresponding to the given ID, we throw an error. You can
    retrieve the accumulated gradients using the
    :meth:`~torch.distributed.autograd.get_gradients` API.

    Arguments:
        context_id (int): The autograd context id for which we should retrieve the gradients.
        roots (list): Tensors which represent the roots of the autograd
                      computation. All the tensors should be scalars.
        retain_graph(bool, optional): If False, the graph used to compute the grad
                      will be freed. Note that in nearly all cases setting this
                      option to True is not needed and often can be worked around
                      in a much more efficient way. Usually, you need to set this
                      to True to run backward multiple times.

    Example::
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     pred = model.forward()
        >>>     loss = loss_func(pred, loss)
        >>>     dist_autograd.backward(context_id, loss)
    """

def get_gradients(context_id: int) -> dict[torch.Tensor, torch.Tensor]:
    """
    get_gradients(context_id: int) -> Dict[Tensor, Tensor]

    Retrieves a map from Tensor to the appropriate gradient for that Tensor
    accumulated in the provided context corresponding to the given ``context_id``
    as part of the distributed autograd backward pass.

    Arguments:
        context_id(int): The autograd context id for which we should retrieve the
                         gradients.

    Returns:
        A map where the key is the Tensor and the value is the associated gradient
        for that Tensor.

    Example::
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = torch.rand((3, 3), requires_grad=True)
        >>>     t2 = torch.rand((3, 3), requires_grad=True)
        >>>     loss = t1 + t2
        >>>     dist_autograd.backward(context_id, [loss.sum()])
        >>>     grads = dist_autograd.get_gradients(context_id)
        >>>     print(grads[t1])
        >>>     print(grads[t2])
    """
