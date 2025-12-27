"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of compiled codes.
  - CodeCacheEntry: Data structure for a single code being compiled by torch.compile.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or host.
"""

import abc
import contextlib
import dataclasses
import functools
import types
from collections.abc import Callable, Generator
from typing import Any, NewType

from torch._dynamo.precompile_context import PrecompileCacheArtifact
from torch.compiler._cache import CacheArtifactFactory

logger = ...

@dataclasses.dataclass(frozen=True)
class SerializedCode:
    """SerializedCode(co_argcount: int, co_posonlyargcount: int, co_kwonlyargcount: int, co_nlocals: int, co_stacksize: int, co_flags: int, co_code: bytes, co_consts: tuple[typing.Any, ...], co_names: tuple[str, ...], co_varnames: tuple[str, ...], co_filename: str, co_name: str, co_firstlineno: int, co_cellvars: tuple[str, ...], co_freevars: tuple[str, ...], co_linetable: Optional[bytes] = None, co_qualname: Optional[str] = None, co_exceptiontable: Optional[bytes] = None, co_lnotab: Optional[str] = None)"""

    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_linetable: bytes | None = ...
    co_qualname: str | None = ...
    co_exceptiontable: bytes | None = ...
    co_lnotab: str | None = ...
    @classmethod
    @functools.cache
    def from_code_object(cls, code: types.CodeType) -> SerializedCode: ...
    @classmethod
    @functools.cache
    def to_code_object(cls, serialized_code: SerializedCode) -> types.CodeType: ...

@dataclasses.dataclass
class _GuardedCodeCacheEntry:
    """
    Contains the serializable information associated with a single compilation in dynamo.
    To restore an execution of compiled code, we will need to serialize the following data:
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards.
    """

    guards_state: bytes
    dynamo_code: SerializedCode

_BackendId = NewType("_BackendId", str)
_FunctionId = NewType("_FunctionId", str)

@dataclasses.dataclass(frozen=True)
class InlinedSource:
    """InlinedSource(module: str, firstlineno: int, lastlineno: int, checksum: str)"""

    module: str
    firstlineno: int
    lastlineno: int
    checksum: str

@dataclasses.dataclass
class DynamoCaptureOutput:
    """Core information generated from Dynamo for fullgraph=True."""

    guarded_codes: list[_GuardedCodeCacheEntry]
    backend_ids: list[_BackendId]

@dataclasses.dataclass
class _DynamoCodeCacheEntry(DynamoCaptureOutput):
    """
    Contains the serializable information associated with a single code object
    in dynamo. To restore an execution of compiled code, we will need the following
    ingredients:
      1. The "original" code object, which serves as the entry point for eager
         execution, i.e. the code only executed when there's no cache entry hit.
      2. The python module name this code object belongs to, for identifying the
         enclosing global scope to inject compiled and resume functions.
      3. A list of function names that pointing to this code object. There could be
         multiple function objects pointing to the same code such as recursive functions.
      4. A list of guarded code that eval frame dispatches to.
      5. A list of imported module objects unioned from all compiled branches.
      6. A list of "backends" (compiled fx graph) unioned from all compield branches.
      7. A string path used to access the original code object users defined.
         A code object can be accessed by "{python_module}.{function_name}.{code_source}" .
      8. A boolean flag indicating whether the function is installed to global scope.
      9. A boolean flag indicating whether the function has a compile id.
      10. Whether or not this code entry was bypassed
    """

    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    import_sources: dict[str, str]
    code_source: str | None
    install_to_global: bool
    has_compile_id: bool = ...
    bypassed: bool = ...

@dataclasses.dataclass
class _DynamoCacheEntry:
    """_DynamoCacheEntry(codes: list[torch._dynamo.package._DynamoCodeCacheEntry], inlined_sources: set[torch._dynamo.package.InlinedSource], python_version: str = '3.13.11', torch_version: str = '2.9.1')"""

    codes: list[_DynamoCodeCacheEntry]
    inlined_sources: set[InlinedSource]
    python_version: str = ...
    torch_version: str = ...
    @property
    def backend_ids(self) -> set[_BackendId]: ...

@CacheArtifactFactory.register
class _DynamoCacheArtifact(PrecompileCacheArtifact[_DynamoCacheEntry]):
    @staticmethod
    def type() -> str: ...
    def after_deserialization(self) -> _DynamoCacheEntry: ...

class CompilePackage:
    """
    CompilePackage is considered a low level component and should not be directly exposed to
    end users. It has the following interface:

    1. `CompilePackage.__init__()` which optionally takes previously serialized dynamo states.
        a. when `dynamo` argument is None, it will construct a brand new CompilePackage object.
        b. when `dynamo` argument is not None, it will load a pre-compiled dynamo state.
    2. `package.save()` which dumps the dynamo and backend states to a DynamoCacheEntry object.
    3. `package.install(backends) which will handle all the side-effectful global scope
        updates with compiled functions and resume functions.
    """
    def __init__(
        self, fn: Callable[..., Any] | None, dynamo: _DynamoCacheEntry | None = ..., ignore_inlined_sources: bool = ...
    ) -> None: ...
    def is_initialized(self) -> bool: ...
    def initialize(
        self, fn: Any, dynamo: _DynamoCacheEntry | None = ..., ignore_inlined_sources: bool = ...
    ) -> None: ...
    @property
    def cached_backends(self) -> dict[_BackendId, Any]: ...
    @functools.cached_property
    def source_id(self) -> str: ...
    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None]: ...
    def add_guarded_code(self, guards_state: bytes, dynamo_code: types.CodeType) -> None: ...
    def add_inlined_source(self, sources: list[types.CodeType]) -> None: ...
    def bypass_current_entry(self) -> None: ...
    def add_resume_function(
        self, python_code: types.CodeType, python_module: str, function_name: str | None
    ) -> None: ...
    def add_import_source(self, alias: str, module_name: str) -> None: ...
    def add_backend_id(self, backend_id: str, backend: Any | None = ...) -> None: ...
    def validate(self) -> None: ...
    def uninstall(self) -> None: ...
    def install(self, backends: dict[_BackendId, Any]) -> None:
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
    def cache_entry(self) -> _DynamoCacheEntry: ...
    @staticmethod
    def source_id_from_fn(fn: Callable[..., Any]) -> str: ...

@CacheArtifactFactory.register
class EagerCacheArtifact(PrecompileCacheArtifact[Any]):
    @staticmethod
    def type() -> str: ...
    def after_deserialization(self) -> Any: ...

type _Backends = dict[_BackendId, PrecompileCacheArtifact[Any]]

class DynamoStore(abc.ABC):
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.

    This is an abstract base class for different storage implementations.
    """
    def record_package(self, package: CompilePackage) -> None:
        """Records a package to PrecompileContext, so that it can be serialized later."""
    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        """Records eager fx graphs to PrecompileContext for testing purposes."""
    @abc.abstractmethod
    def clear(self) -> None: ...
    @abc.abstractmethod
    def write(self, dynamo: _DynamoCacheEntry, backends: _Backends, path: str) -> None:
        """
        Abstract method to write dynamo cache entry and backends to storage.

        Args:
            dynamo: The dynamo cache entry to write
            backends: Dictionary of backend content to write
            path: Path or key to identify where to write the data
        """
        ...
    def save_cache_entry(self, cache_entry: _DynamoCacheEntry, key: str) -> None:
        """Saves a package to a given path. Grabs backends from PrecompileContext."""
    def save_package(self, package: CompilePackage, key: str) -> None:
        """Saves a package to a given path. Grabs backends from PrecompileContext."""
    @abc.abstractmethod
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]:
        """
        Abstract method to read dynamo cache entry and backends from storage.

        Args:
            path: Path or key to identify where to read the data from

        Returns:
            A tuple containing (dynamo_cache_entry, backend_content)
        """
        ...
    def load_cache_entry(self, key: str) -> tuple[_DynamoCacheEntry, dict[_BackendId, Any]]: ...
    def load_package(self, fn: Any, key: str) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        """Loads a package from a given path and returns it plus a list of deserialized backends"""

class InMemoryDynamoStore(DynamoStore):
    """A DynamoStore implementation that keeps state about CompilePackages in memory."""
    def __init__(self) -> None: ...
    def clear(self) -> None: ...
    def write(self, dynamo: _DynamoCacheEntry, backends: _Backends, path: str) -> None:
        """Store the dynamo cache entry and backends in memory instead of writing to disk."""
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]:
        """Read dynamo cache entry and backends from memory."""

class DiskDynamoStore(DynamoStore):
    """A DynamoStore implementation that keeps state about CompilePackages on disk."""
    def __init__(self, path_prefix: str = ...) -> None:
        """
        Initialize a DiskDynamoStore with a path prefix.

        Args:
            path_prefix: Prefix directory for where to put CompilePackages on disk
        """
    def clear(self) -> None:
        """Clear all CompilePackages from disk."""
    def write(self, dynamo: _DynamoCacheEntry, backends: _Backends, path: str) -> None:
        """Write dynamo cache entry and backends to disk."""
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]:
        """Read dynamo cache entry and backends from disk."""

class DiskDynamoCache(DiskDynamoStore):
    """
    Special DiskDynamoStore which adds some helper functions for automatically
    tracking paths of packages
    """
    def save(self, package: CompilePackage) -> None:
        """Saves a package to a given path. Grabs backends from PrecompileContext."""
    def load(self, fn: Callable[..., Any]) -> tuple[_DynamoCacheEntry, dict[_BackendId, Any]] | None:
        """Loads a package from a given path and returns it plus a list of deserialized backends"""
    def load_and_install_package(self, fn: Callable[..., Any]) -> CompilePackage | None:
        """Load directly into a package and install backends"""

DynamoCache = ...
