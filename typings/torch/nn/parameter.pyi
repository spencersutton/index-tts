from typing import TypeIs

from torch import Tensor, device, dtype

class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...

def is_lazy(param: Tensor) -> TypeIs[UninitializedParameter | UninitializedBuffer]:
    """
    Returns whether ``param`` is an ``UninitializedParameter`` or ``UninitializedBuffer``.

    Args:
        param (Any): the input to check.
    """

class UninitializedParameter(Tensor):
    """
    A parameter that is not initialized.

    Uninitialized Parameters are a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    def materialize(self, shape: tuple[int, ...], device: device | None = ..., dtype: dtype | None = ...) -> None:
        """
        Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """

class Buffer(Tensor):
    """
    A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

    Buffers are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s -- when they're
    assigned as Module attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~torch.nn.Module.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~torch.nn.Module.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """

    persistent: bool
    def __init__(self, data: Tensor = ..., requires_grad: bool = ..., persistent: bool = ...) -> None: ...

class UninitializedBuffer(Tensor):
    """
    A buffer that is not initialized.

    Uninitialized Buffer is a a special case of :class:`torch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    persistent: bool
    def __init__(self, data: Tensor = ..., requires_grad: bool = ..., persistent: bool = ...) -> None: ...
    def materialize(self, shape: tuple[int, ...], device: device | None = ..., dtype: dtype | None = ...) -> None:
        """
        Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
