from abc import abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, override

from torch.compiler._cache import CacheArtifact, CacheArtifactManager, CacheArtifactsResult, CacheInfo
from torch.utils._appending_byte_serializer import AppendingByteSerializer
from torch.utils._ordered_set import OrderedSet

T = TypeVar("T")
logger = ...

class PrecompileCacheArtifact[T](CacheArtifact):
    """
    Data for each cache artifact that will be serialized and deserialized by
    PrecompileContext, rather than CacheArtifactManager.
    T represents the deserialized type of the artifact, i.e. the return type of after_deserialization

    PrecompileCacheArtifact is a frozen dataclass - you can add new serializable fields and metadata specific to your own artifacts
    as needed, and use them in after_deserialization.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """
    @override
    def populate_cache(self) -> None: ...
    @override
    def precompile_compatible(self) -> bool: ...
    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

class EditablePrecompileCacheArtifact[T]:
    """A PrecompileCacheArtifact whose content isn't encoded until we call PrecompileContext.serialize()"""
    def __init__(self, artifact_type: str, content: Any, key: str) -> None: ...
    def real_encode(self) -> PrecompileCacheArtifact[T]:
        """Actually encode the object"""
    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """Edit the content of an existing artifact"""

class PrecompileContext(CacheArtifactManager):
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact
     - DynamoCodeStateArtifact
     - AutotuneCacheArtifact (regular autotune results, same as Megacache)
    """

    _new_cache_artifacts_by_key: dict[str, EditablePrecompileCacheArtifact[object] | CacheArtifact] = ...
    _new_cache_artifacts: CacheArtifactsResult = ...
    _seen_artifacts: OrderedSet[CacheArtifact] = ...
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]] = ...
    _cache_info: CacheInfo = ...
    @classmethod
    def clear(cls) -> None: ...
    @override
    @classmethod
    def record_artifact(cls, artifact_type: str, key: str, content: Any, editable: bool = ...) -> None:
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """Edit the content of an existing artifact"""
    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> CacheArtifact | None:
        """Serialize all artifacts with the given key returned in a list."""
    @classmethod
    def serialize(cls) -> tuple[bytes, CacheInfo] | None: ...
    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo: ...
