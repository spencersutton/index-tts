import torch.fx.graph
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Optional, Union
from collections.abc import Callable
from typing import TypeAlias

class CustomGraphPass(ABC):
    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None: ...
    @abstractmethod
    def uuid(self) -> Any | None: ...

class CustomGraphModulePass(ABC):
    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None: ...
    @abstractmethod
    def uuid(self) -> Any | None: ...

type CustomGraphPassType = CustomGraphPass | Callable[[torch.fx.graph.Graph], None] | None

@lru_cache(1)
def get_hash_for_files(paths: tuple[str], extra: str = ...) -> bytes: ...

class CustomPartitionerFn(ABC):
    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]: ...
    @abstractmethod
    def uuid(self) -> Any | None: ...

type CustomPartitionerFnType = CustomPartitionerFn | None
