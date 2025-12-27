"""
This module provides utilities for analyzing and optimizing Python bytecode.
Key functionality includes:
- Dead code elimination
- Jump instruction optimization
- Stack size analysis and verification
- Live variable analysis
- Line number propagation and cleanup
- Exception table handling for Python 3.11+

The utilities in this module are used to analyze and transform bytecode
for better performance while maintaining correct semantics.
"""

import dataclasses
import sys
from typing import Any

from .bytecode_transformation import Instruction

TERMINAL_OPCODES = ...
if (3, 12) <= sys.version_info < (3, 14): ...
JUMP_OPCODES = ...
JUMP_OPNAMES = ...
HASLOCAL = ...
HASFREE = ...
stack_effect = ...

def get_indexof(insts: list[Instruction]) -> dict[Instruction, int]:
    """
    Get a mapping from instruction memory address to index in instruction list.
    Additionally checks that each instruction only appears once in the list.
    """

def remove_dead_code(instructions: list[Instruction]) -> list[Instruction]:
    """Dead code elimination"""

def remove_pointless_jumps(instructions: list[Instruction]) -> list[Instruction]:
    """Eliminate jumps to the next instruction"""

def propagate_line_nums(instructions: list[Instruction]) -> None:
    """Ensure every instruction has line number set in case some are removed"""

def remove_extra_line_nums(instructions: list[Instruction]) -> None:
    """Remove extra starts line properties before packing bytecode"""

@dataclasses.dataclass
class ReadsWrites:
    """ReadsWrites(reads: set[typing.Any], writes: set[typing.Any], visited: set[typing.Any])"""

    reads: set[Any]
    writes: set[Any]
    visited: set[Any]

def livevars_analysis(instructions: list[Instruction], instruction: Instruction) -> set[Any]: ...

@dataclasses.dataclass
class FixedPointBox:
    """FixedPointBox(value: bool = True)"""

    value: bool = ...

@dataclasses.dataclass
class StackSize:
    """StackSize(low: Union[int, float], high: Union[int, float], fixed_point: torch._dynamo.bytecode_analysis.FixedPointBox)"""

    low: int | float
    high: int | float
    fixed_point: FixedPointBox
    def zero(self) -> None: ...
    def offset_of(self, other: StackSize, n: int) -> None: ...
    def exn_tab_jump(self, depth: int) -> None: ...

def stacksize_analysis(instructions: list[Instruction]) -> int | float: ...
