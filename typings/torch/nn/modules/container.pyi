from collections import OrderedDict, abc as container_abcs
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Self, TypeVar, overload
from warnings import deprecated

from torch import Tensor

from indextts.util import patch_call

from .module import Module

__all__ = ["Container", "ModuleDict", "ModuleList", "ParameterDict", "ParameterList", "Sequential"]
T = TypeVar("T", bound=Module)

@deprecated(
    "`nn.Container` is deprecated. All of it's functionality is now implemented in `nn.Module`. Subclass that instead.",
    category=FutureWarning,
)
class Container(Module):
    def __init__(self, **kwargs: Any) -> None: ...

class Sequential[T: Module](Module):
    """
    A sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU()
        )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 20, 5)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(20, 64, 5)),
                    ("relu2", nn.ReLU()),
                ]
            )
        )
    """

    _modules: dict[str, T]
    @overload
    def __init__(self, *args: T) -> None: ...
    @overload
    def __init__(self, arg: OrderedDict[str, T]) -> None: ...
    def __init__(self, *args) -> None: ...
    def __getitem__(self, idx: slice | int) -> T: ...
    def __setitem__(self, idx: int, module: T) -> None: ...
    def __delitem__(self, idx: slice | int) -> None: ...
    def __len__(self) -> int: ...
    def __add__(self, other) -> Sequential[T]: ...
    def pop(self, key: int | slice) -> T:
        """Pop ``key`` from self."""
    def __iadd__(self, other) -> Self: ...
    def __mul__(self, other: int) -> Sequential[T]: ...
    def __rmul__(self, other: int) -> Sequential[T]: ...
    def __imul__(self, other: int) -> Self: ...
    def __dir__(self) -> list[str]: ...
    def __iter__(self) -> Iterator[T]: ...
    def append(self, module: T) -> Self:
        """
        Append a given module to the end.

        Args:
            module (nn.Module): module to append

        Example::

            >>> import torch.nn as nn
            >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
            >>> n.append(nn.Linear(3, 4))
            Sequential(
                (0): Linear(in_features=1, out_features=2, bias=True)
                (1): Linear(in_features=2, out_features=3, bias=True)
                (2): Linear(in_features=3, out_features=4, bias=True)
            )
        """
    def insert(self, index: int, module: T) -> Self:
        """
        Inserts a module into the Sequential container at the specified index.

        Args:
            index (int): The index to insert the module.
            module (Module): The module to be inserted.

        Example::

            >>> import torch.nn as nn
            >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
            >>> n.insert(0, nn.Linear(3, 4))
            Sequential(
                (0): Linear(in_features=3, out_features=4, bias=True)
                (1): Linear(in_features=1, out_features=2, bias=True)
                (2): Linear(in_features=2, out_features=3, bias=True)
            )
        """
    def extend(self, sequential: Iterable[T]) -> Self:
        """
        Extends the current Sequential container with layers from another Sequential container.

        Args:
            sequential (Sequential): A Sequential container whose layers will be added to the current container.

        Example::

            >>> import torch.nn as nn
            >>> n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
            >>> other = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5))
            >>> n.extend(other) # or `n + other`
            Sequential(
                (0): Linear(in_features=1, out_features=2, bias=True)
                (1): Linear(in_features=2, out_features=3, bias=True)
                (2): Linear(in_features=3, out_features=4, bias=True)
                (3): Linear(in_features=4, out_features=5, bias=True)
            )
        """
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Runs the forward pass."""
    @patch_call(forward)
    def __call__(self) -> None: ...

class ModuleList[T: Module = Module](Module):
    """
    Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    _modules: dict[str, T]
    def __init__(self, modules: Iterable[T] | None = ...) -> None: ...
    @overload
    def __getitem__(self, idx: slice) -> ModuleList[T]: ...
    @overload
    def __getitem__(self, idx: int) -> T: ...
    def __getitem__(self, idx: int | slice) -> T | ModuleList[T]: ...
    def __setitem__(self, idx: int, module: T) -> None: ...
    def __delitem__(self, idx: int | slice) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __iadd__(self, modules: Iterable[T]) -> Self: ...
    def __add__(self, other: Iterable[T]) -> ModuleList[T]: ...
    def __dir__(self) -> list[str]: ...
    def insert(self, index: int, module: T) -> None:
        """
        Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
    def append(self, module: T) -> Self:
        """
        Append a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
    def pop(self, key: int | slice) -> T: ...
    def extend(self, modules: Iterable[T]) -> Self:
        """
        Append modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """

class ModuleDict[T: Module](Module):
    """
    Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.choices = nn.ModuleDict(
                    {"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)}
                )
                self.activations = nn.ModuleDict(
                    [["lrelu", nn.LeakyReLU()], ["prelu", nn.PReLU()]]
                )

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    _modules: dict[str, T]
    def __init__(self, modules: Mapping[str, T] | None = ...) -> None: ...
    def __getitem__(self, key: str) -> T: ...
    def __setitem__(self, key: str, module: T) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __contains__(self, key: str) -> bool: ...
    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
    def pop(self, key: str) -> T:
        """
        Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
    def keys(self) -> container_abcs.KeysView[str]:
        """Return an iterable of the ModuleDict keys."""
    def items(self) -> container_abcs.ItemsView[str, T]:
        """Return an iterable of the ModuleDict key/value pairs."""
    def values(self) -> container_abcs.ValuesView[T]:
        """Return an iterable of the ModuleDict values."""
    def update(self, modules: Mapping[str, T]) -> None:
        """
        Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """

class ParameterList[T](Module):
    """
    Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be used like a regular Python
    list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered,
    and will be visible by all :class:`~torch.nn.Module` methods.

    Note that the constructor, assigning an element of the list, the
    :meth:`~torch.nn.ParameterList.append` method and the :meth:`~torch.nn.ParameterList.extend`
    method will convert any :class:`~torch.Tensor` into :class:`~torch.nn.Parameter`.

    Args:
        parameters (iterable, optional): an iterable of elements to add to the list.

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.params = nn.ParameterList(
                    [nn.Parameter(torch.randn(10, 10)) for i in range(10)]
                )

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """
    def __init__(self, values: Iterable[Any] | None = ...) -> None: ...
    @overload
    def __getitem__(self, idx: int) -> Any: ...
    @overload
    def __getitem__(self: T, idx: slice) -> T: ...
    def __getitem__(self, idx) -> Self | Any: ...
    def __setitem__(self, idx: int, param: Any) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __iadd__(self, parameters: Iterable[Any]) -> Self: ...
    def __dir__(self) -> list[str]: ...
    def append(self, value: Any) -> Self:
        """
        Append a given value at the end of the list.

        Args:
            value (Any): value to append
        """
    def extend(self, values: Iterable[Any]) -> Self:
        """
        Append values from a Python iterable to the end of the list.

        Args:
            values (iterable): iterable of values to append
        """
    def extra_repr(self) -> str:
        """Return the extra representation of the module."""
    def __call__(self, *args, **kwargs): ...

class ParameterDict(Module):
    """
    Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Module methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
    :class:`~torch.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.params = nn.ParameterDict(
                    {
                        "left": nn.Parameter(torch.randn(5, 10)),
                        "right": nn.Parameter(torch.randn(5, 10)),
                    }
                )

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """
    def __init__(self, parameters: Any = ...) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __reversed__(self) -> Iterator[str]: ...
    def copy(self) -> ParameterDict:
        """Return a copy of this :class:`~torch.nn.ParameterDict` instance."""
    def __contains__(self, key: str) -> bool: ...
    def setdefault(self, key: str, default: Any | None = ...) -> Any:
        """
        Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
    def pop(self, key: str) -> Any:
        """
        Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
    def popitem(self) -> tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
    def get(self, key: str, default: Any | None = ...) -> Any:
        """
        Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
    def fromkeys(self, keys: Iterable[str], default: Any | None = ...) -> ParameterDict:
        """
        Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
    def keys(self) -> container_abcs.KeysView[str]:
        """Return an iterable of the ParameterDict keys."""
    def items(self) -> Iterable[tuple[str, Any]]:
        """Return an iterable of the ParameterDict key/value pairs."""
    def values(self) -> Iterable[Any]:
        """Return an iterable of the ParameterDict values."""
    def update(self, parameters: Mapping[str, Any] | ParameterDict) -> None:
        """
        Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
    def extra_repr(self) -> str: ...
    def __call__(self, input): ...
    def __or__(self, other: ParameterDict) -> ParameterDict: ...
    def __ror__(self, other: ParameterDict) -> ParameterDict: ...
    def __ior__(self, other: ParameterDict) -> Self: ...
