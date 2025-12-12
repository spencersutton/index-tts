import dataclasses
from typing import Callable, Optional
from torch._guards import GuardsSet
from .types import GuardFail, GuardFilterEntry

"""Hook system for Dynamo's guard functionality.

This module provides a way to register callback functions that are triggered during
guard-related operations.

The Hooks class manages two types of hook functions:
- guard_export_fn: Called when guards need to be exported, taking a GuardsSet as input
- guard_fail_fn: Called when a guard check fails, taking a GuardFail object as input
These hooks enable customization of guard export and failure handling behaviors.
"""

@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[GuardsSet], None]] = ...
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = ...
    guard_filter_fn: Optional[Callable[[list[GuardFilterEntry]], list[bool]]] = ...
