import dataclasses
from collections.abc import Callable

from torch._guards import GuardsSet

from .types import GuardFail, GuardFilterEntry

@dataclasses.dataclass
class Hooks:
    guard_export_fn: Callable[[GuardsSet], None] | None = ...
    guard_fail_fn: Callable[[GuardFail], None] | None = ...
    guard_filter_fn: Callable[[list[GuardFilterEntry]], list[bool]] | None = ...
