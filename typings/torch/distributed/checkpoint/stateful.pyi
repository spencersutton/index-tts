from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = ["Stateful", "StatefulT"]

@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

StatefulT = TypeVar("StatefulT", bound=Stateful)
