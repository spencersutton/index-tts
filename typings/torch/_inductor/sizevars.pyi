from collections.abc import Callable, Iterable, Sequence
from typing import Any

import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .utils import VarRanges
from .virtualized import V

log = ...

def statically_known_true(
    shape_env: ShapeEnv,
    expr: sympy.Basic | bool,
    axioms: tuple[sympy.Expr] | None = ...,
    var_to_range: tuple[tuple[sympy.Symbol, ValueRanges[Any]]] | None = ...,
) -> bool: ...

class SizeVarAllocator:
    """
    A class that manages symbolic size variables and their relationships.

    This class works with the ShapeEnv to handle symbolic shape expressions,
    simplify them, and provide utilities for guarding, checking, and evaluating
    symbolic expressions. It also manages precomputed replacements and stride
    calculations for tensor operations.
    """
    def __init__(self, shape_env=...) -> None: ...
    def simplify(self, expr: Expr): ...
    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        """self._simplify_with_ranges() can be expensive, cache its results"""
    def make_simplify_loops_cache(self):
        """self._simplify_with_ranges() can be expensive, cache its results"""
    def statically_known_true(self, expr: sympy.Basic | bool) -> bool:
        """
        Returns true if an expression is always true (symbolically or via guards),
        false otherwise. Never add guards, or throw data dependent errors.
        """
    def statically_known_equals(self, left: Expr | int, right: Expr | int) -> bool:
        """Returns a bool indicating if it is sound to optimize as if left and right are equal."""
    def statically_known_list_equals(self, left: Sequence[Expr], right: Sequence[Expr]) -> bool:
        """Returns a bool indicating if it is sound to optimize as if left and right lists are equal."""
    def statically_known_leq(self, left: Expr, right: Expr | int) -> bool:
        """Returns a bool indicating if it is sound to optimize as if left is less than or equal to right."""
    def statically_known_geq(self, left: Expr, right: Expr | int) -> bool:
        """Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right."""
    def statically_known_lt(self, left: Expr, right: Expr | int) -> bool:
        """Returns a bool indicating if it is sound to optimize as if left is less than right."""
    def statically_known_gt(self, left: Expr, right: Expr | int) -> bool:
        """Returns a bool indicating if it is sound to optimize as if left is greater than right."""
    def statically_known_multiple_of(self, numerator: Expr, denominator: Expr | int) -> bool:
        """Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator."""
    def statically_known_power_of_2(self, expr: Expr) -> bool:
        """Returns a bool indicating if x is known to be a power of 2."""
    def expect_true(self, expr: Expr) -> bool:
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike check, this WON'T raise an error if expr isn't actually true.
        check Note [expect_true].
        """
    def check(self, expr: Expr) -> None:
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike expect_true, this WILL raise an error if expr isn't actually true.
        check Note [expect_true].
        """
    def check_equals(self, left: Expr, right: Expr) -> None:
        """check(sympy.Eq(left, right))."""
    def check_equals_and_simplify(self, left: Expr, right: Expr) -> Expr:
        """
        check(sympy.Eq(left, right)) and returns left after applying
        inv_precomputed_replacements.
        """
    def check_leq(self, left: Expr, right: Expr) -> None: ...
    def check_lt(self, left: Expr, right: Expr) -> None: ...
    def guard_or_false(self, left): ...
    def guard_or_true(self, left): ...
    def evaluate_expr(
        self, left: Expr | sympy.logic.boolalg.Boolean, size_oblivious: bool = ..., fallback_value: bool | None = ...
    ) -> bool: ...
    def is_size_one_or_false(self, size: Expr) -> bool:
        """
        Return True if size equals 1.

        Unbacked symbolic sizes return False without introducing a guard.
        """
    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        """return the smaller of left and right, and guard on that choice"""
    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        """return the larger of left and right, and guard on that choice"""
    def guard_int(self, expr: Expr | int) -> int:
        """
        Similar to guard_int in symbolic_shapes.py, except this function works with SymPy
        expressions instead of SymNodes. It extracts the value represented by expr from shapeEnv
        and specialize the compiled graph on it. Raises an error if the result cannot be
        determined due to unhinted or unbacked symbols.
        """
    def guard_int_seq(self, left: Sequence[Expr | int]) -> list[int]:
        """Apply guard_int on a sequence of inputs."""
    def remove_precomputed_replacements(self, expr: Expr) -> Expr: ...
    def symbolic_hint(self, expr: Expr | int, hint_override: int | None = ...) -> Expr | int: ...
    def size_hint(self, expr: Expr | int, *, fallback: int | None = ..., hint_override: int | None = ...) -> int: ...
    def size_hint_or_throw(self, expr: Expr | int) -> int: ...
    def size_hints(
        self, exprs: Iterable[Expr | int], *, fallback: int | None = ..., hint_override: int | None = ...
    ) -> tuple[int, ...]: ...
    def size_hints_or_throw(self, exprs: Iterable[Expr | int]) -> tuple[int, ...]: ...
    def make_stride_vars_cache(self): ...
    def atomically_apply_size_hint(self, expr: Expr | int, *, fallback: int | None = ...) -> Expr | int: ...
    def offset_var(self, index: Expr, vars: Sequence[sympy.Symbol]) -> Expr:
        """Extract offset part of an indexing expression"""
    def stride_hints(
        self, index: Expr, vars: Sequence[sympy.Symbol], support_vars: Sequence[sympy.Symbol] | None = ...
    ) -> list[int]: ...
    def stride_order(self, index: Expr, vars: list[sympy.Symbol]) -> list[int]: ...
    def lookup_precomputed_size(self, expr: Expr) -> Expr: ...
    def free_symbols(self) -> OrderedSet[sympy.Symbol]: ...
    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        """
        A pair of special ModularIndexing can be combined.

        E.g. ModularIndexing(ModularIndexing(x, 1, a), 1, b)
        We can simplify this to ModuleIndexing(x, 1, b), if
        1. x is non negative integer
        2. a and b are positive integers
        3. a is a multiple of b.
        """
    def expand_floor_div(self, index: sympy.Expr) -> bool | tuple[sympy.Expr, sympy.Expr]:
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplified.

        E.g., for a 2D contiguous tensor with shape [a, 2 * b], and index variables
        x1, x2, index expression 'x1 * 2b + x2' can be easily combined.
        But index expression 'x1 * b + x2 // 2' can not.
        By expanding the FloorDiv to the entire expression, we get
        '(x1 * 2b + x2) // 2'. This transformation allows us to merge loops
        for the numerator!

        Return false if this optimization can be applied;
        Return the new expression and the denominator otherwise.
        The original expression will be equivalent to 'new_expression // denominator'
        """

def join_dimensions(expr: Expr) -> Expr: ...

class SimplifyIndexing(V.WrapperHandler):
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """
    def __init__(self, inner, var_ranges: VarRanges) -> None: ...
    def load(self, name: str, index: sympy.Expr): ...
    def store(self, name, index, value, mode=...): ...
    def store_reduction(self, name, index, value): ...
    def index_expr(self, index, dtype): ...
    def check_bounds(self, index, size, lower, upper): ...
