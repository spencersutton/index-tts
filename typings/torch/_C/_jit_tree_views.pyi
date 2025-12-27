from typing import Any

class SourceRange:
    def highlight(self) -> str:
        """highlight(self: torch._C._jit_tree_views.SourceRange) -> str"""
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...

class SourceRangeFactory:
    def __init__(self, text: str, filename: Any, file_lineno: int, leading_whitespace_chars: int) -> None:
        """__init__(self: torch._C._jit_tree_views.SourceRangeFactory, arg0: str, arg1: object, arg2: typing.SupportsInt, arg3: typing.SupportsInt) -> None"""
    def make_range(self, line: int, start_col: int, end_col: int) -> SourceRange:
        """make_range(self: torch._C._jit_tree_views.SourceRangeFactory, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> torch._C._jit_tree_views.SourceRange"""
    def make_raw_range(self, start: int, end: int) -> SourceRange:
        """make_raw_range(self: torch._C._jit_tree_views.SourceRangeFactory, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> torch._C._jit_tree_views.SourceRange"""
    @property
    def source(self) -> str: ...

class TreeView:
    def range(self) -> SourceRange:
        """range(self: torch._C._jit_tree_views.TreeView) -> torch._C._jit_tree_views.SourceRange"""
    def dump(self) -> None:
        """dump(self: torch._C._jit_tree_views.TreeView) -> None"""

class Ident(TreeView):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__(self: torch._C._jit_tree_views.Ident, arg0: torch._C._jit_tree_views.SourceRange, arg1: str) -> None"""
    @property
    def name(self) -> str: ...

class Param(TreeView):
    def __init__(self, type: Any | None, name: Ident, kwarg_only: bool) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C._jit_tree_views.Param, arg0: torch::jit::Expr, arg1: torch._C._jit_tree_views.Ident, arg2: bool) -> None

        2. __init__(self: torch._C._jit_tree_views.Param, arg0: torch::jit::Maybe<torch::jit::Expr>, arg1: torch._C._jit_tree_views.Ident, arg2: bool) -> None
        """

class Attribute(TreeView):
    def __init__(self, name: Ident, value: Any) -> None:
        """__init__(self: torch._C._jit_tree_views.Attribute, arg0: torch._C._jit_tree_views.Ident, arg1: torch::jit::Expr) -> None"""

def TrueLiteral(range: SourceRange) -> Any:
    """TrueLiteral(arg0: torch._C._jit_tree_views.SourceRange) -> torch::jit::Expr"""

def FalseLiteral(range: SourceRange) -> Any:
    """FalseLiteral(arg0: torch._C._jit_tree_views.SourceRange) -> torch::jit::Expr"""

def NoneLiteral(range: SourceRange) -> Any:
    """NoneLiteral(arg0: torch._C._jit_tree_views.SourceRange) -> torch::jit::Expr"""

class Stmt(TreeView):
    def __init__(self, thing: TreeView) -> None:
        """__init__(self: torch._C._jit_tree_views.Stmt, arg0: torch._C._jit_tree_views.TreeView) -> None"""

class Expr(TreeView): ...

class Def(TreeView):
    def __init__(self, name: Ident, decl: Any, body: list[Stmt]) -> None:
        """__init__(self: torch._C._jit_tree_views.Def, arg0: torch._C._jit_tree_views.Ident, arg1: torch::jit::Decl, arg2: collections.abc.Sequence[torch._C._jit_tree_views.Stmt]) -> None"""
    def decl(self) -> Any:
        """decl(self: torch._C._jit_tree_views.Def) -> torch::jit::Decl"""
    def name(self) -> Ident:
        """name(self: torch._C._jit_tree_views.Def) -> torch._C._jit_tree_views.Ident"""

class Property(TreeView):
    def __init__(self, r: SourceRange, name: Ident, getter: Def, setter: Def | None) -> None:
        """__init__(self: torch._C._jit_tree_views.Property, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Ident, arg2: torch._C._jit_tree_views.Def, arg3: torch._C._jit_tree_views.Def) -> None"""
    def name(self) -> Ident:
        """name(self: torch._C._jit_tree_views.Property) -> torch._C._jit_tree_views.Ident"""
    def getter_name(self) -> str:
        """getter_name(self: torch._C._jit_tree_views.Property) -> torch._C._jit_tree_views.Ident"""
    def setter_name(self) -> Ident | None:
        """setter_name(self: torch._C._jit_tree_views.Property) -> torch._C._jit_tree_views.Ident | None"""

class ClassDef(TreeView):
    def __init__(self, name: Ident, body: list[Stmt], props: list[Property], assigns: list[Any]) -> None:
        """__init__(self: torch._C._jit_tree_views.ClassDef, arg0: torch._C._jit_tree_views.Ident, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Stmt], arg2: collections.abc.Sequence[torch._C._jit_tree_views.Property], arg3: collections.abc.Sequence[torch::jit::Assign]) -> None"""

class Decl(TreeView):
    def __init__(self, r: SourceRange, params: list[Param], return_type: Expr | None) -> None:
        """__init__(self: torch._C._jit_tree_views.Decl, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Param], arg2: torch._C._jit_tree_views.Expr) -> None"""

class Delete(Stmt):
    def __init__(self, range: SourceRange, targets: list[Expr]) -> None:
        """__init__(self: torch._C._jit_tree_views.Delete, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr]) -> None"""

class WithItem(Expr):
    def __init__(self, range: SourceRange, target: Expr, var: Any | None) -> None:
        """__init__(self: torch._C._jit_tree_views.WithItem, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: torch::jit::Var) -> None"""

class Assign(Stmt):
    def __init__(self, lhs: list[Expr], rhs: Expr, type: Expr | None = ...) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C._jit_tree_views.Assign, arg0: collections.abc.Sequence[torch._C._jit_tree_views.Expr], arg1: torch._C._jit_tree_views.Expr) -> None

        2. __init__(self: torch._C._jit_tree_views.Assign, arg0: collections.abc.Sequence[torch._C._jit_tree_views.Expr], arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr) -> None
        """

class AugAssign(Stmt):
    def __init__(self, lhs: Expr, kind_str: str, rhs: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.AugAssign, arg0: torch._C._jit_tree_views.Expr, arg1: str, arg2: torch._C._jit_tree_views.Expr) -> None"""

class Return(Stmt):
    def __init__(self, range: SourceRange, value: Expr | None) -> None:
        """__init__(self: torch._C._jit_tree_views.Return, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr) -> None"""

class Raise(Stmt):
    def __init__(self, range: SourceRange, expr: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.Raise, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr) -> None"""

class Assert(Stmt):
    def __init__(self, range: SourceRange, test: Expr, msg: Expr | None) -> None:
        """__init__(self: torch._C._jit_tree_views.Assert, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr) -> None"""

class Pass(Stmt):
    def __init__(self, range: SourceRange) -> None:
        """__init__(self: torch._C._jit_tree_views.Pass, arg0: torch._C._jit_tree_views.SourceRange) -> None"""

class Break(Stmt): ...
class Continue(Stmt): ...

class Dots(Expr, TreeView):
    def __init__(self, range: SourceRange) -> None:
        """__init__(self: torch._C._jit_tree_views.Dots, arg0: torch._C._jit_tree_views.SourceRange) -> None"""

class If(Stmt):
    def __init__(self, range: SourceRange, cond: Expr, true_branch: list[Stmt], false_branch: list[Stmt]) -> None:
        """__init__(self: torch._C._jit_tree_views.If, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: collections.abc.Sequence[torch._C._jit_tree_views.Stmt], arg3: collections.abc.Sequence[torch._C._jit_tree_views.Stmt]) -> None"""

class While(Stmt):
    def __init__(self, range: SourceRange, cond: Expr, body: list[Stmt]) -> None:
        """__init__(self: torch._C._jit_tree_views.While, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: collections.abc.Sequence[torch._C._jit_tree_views.Stmt]) -> None"""

class With(Stmt):
    def __init__(self, range: SourceRange, targets: list[WithItem], body: list[Stmt]) -> None:
        """__init__(self: torch._C._jit_tree_views.With, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.WithItem], arg2: collections.abc.Sequence[torch._C._jit_tree_views.Stmt]) -> None"""

class For(Stmt):
    def __init__(self, range: SourceRange, targets: list[Expr], itrs: list[Expr], body: list[Stmt]) -> None:
        """__init__(self: torch._C._jit_tree_views.For, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr], arg2: collections.abc.Sequence[torch._C._jit_tree_views.Expr], arg3: collections.abc.Sequence[torch._C._jit_tree_views.Stmt]) -> None"""

class ExprStmt(Stmt):
    def __init__(self, expr: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.ExprStmt, arg0: torch._C._jit_tree_views.Expr) -> None"""

class Var(Expr):
    def __init__(self, name: Ident) -> None:
        """__init__(self: torch._C._jit_tree_views.Var, arg0: torch._C._jit_tree_views.Ident) -> None"""
    @property
    def name(self) -> str: ...

class BinOp(Expr):
    def __init__(self, kind: str, lhs: Expr, rhs: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.BinOp, arg0: str, arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr) -> None"""

class UnaryOp(Expr):
    def __init__(self, range: SourceRange, kind: str, expr: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.UnaryOp, arg0: torch._C._jit_tree_views.SourceRange, arg1: str, arg2: torch._C._jit_tree_views.Expr) -> None"""

class Const(Expr):
    def __init__(self, range: SourceRange, value: str) -> None:
        """__init__(self: torch._C._jit_tree_views.Const, arg0: torch._C._jit_tree_views.SourceRange, arg1: str) -> None"""

class StringLiteral(Expr):
    def __init__(self, range: SourceRange, value: str) -> None:
        """__init__(self: torch._C._jit_tree_views.StringLiteral, arg0: torch._C._jit_tree_views.SourceRange, arg1: str) -> None"""

class Apply(Expr):
    def __init__(self, expr: Expr, args: list[Expr], kwargs: list[Attribute]) -> None:
        """__init__(self: torch._C._jit_tree_views.Apply, arg0: torch._C._jit_tree_views.Expr, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr], arg2: collections.abc.Sequence[torch._C._jit_tree_views.Attribute]) -> None"""

class Select(Expr):
    def __init__(self, expr: Expr, field: Ident) -> None:
        """__init__(self: torch._C._jit_tree_views.Select, arg0: torch._C._jit_tree_views.Expr, arg1: torch._C._jit_tree_views.Ident) -> None"""

class TernaryIf(Expr):
    def __init__(self, cond: Expr, true_expr: Expr, false_expr: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.TernaryIf, arg0: torch._C._jit_tree_views.Expr, arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr) -> None"""

class ListComp(Expr):
    def __init__(self, range: SourceRange, elt: Expr, target: Expr, iter: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.ListComp, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr, arg3: torch._C._jit_tree_views.Expr) -> None"""

class DictComp(Expr):
    def __init__(self, range: SourceRange, key: Expr, value: Expr, target: Expr, iter: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.DictComp, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr, arg3: torch._C._jit_tree_views.Expr, arg4: torch._C._jit_tree_views.Expr) -> None"""

class ListLiteral(Expr):
    def __init__(self, range: SourceRange, args: list[Expr]) -> None:
        """__init__(self: torch._C._jit_tree_views.ListLiteral, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr]) -> None"""

class TupleLiteral(Expr):
    def __init__(self, range: SourceRange, args: list[Expr]) -> None:
        """__init__(self: torch._C._jit_tree_views.TupleLiteral, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr]) -> None"""

class DictLiteral(Expr):
    def __init__(self, range: SourceRange, keys: list[Expr], values: list[Expr]) -> None:
        """__init__(self: torch._C._jit_tree_views.DictLiteral, arg0: torch._C._jit_tree_views.SourceRange, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr], arg2: collections.abc.Sequence[torch._C._jit_tree_views.Expr]) -> None"""

class Subscript(Expr):
    def __init__(self, base: Expr, subscript_exprs: list[Expr]) -> None:
        """__init__(self: torch._C._jit_tree_views.Subscript, arg0: torch._C._jit_tree_views.Expr, arg1: collections.abc.Sequence[torch._C._jit_tree_views.Expr]) -> None"""

class SliceExpr(Expr):
    def __init__(self, range: SourceRange, lower: Expr | None, upper: Expr | None, step: Expr | None) -> None:
        """__init__(self: torch._C._jit_tree_views.SliceExpr, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr, arg2: torch._C._jit_tree_views.Expr, arg3: torch._C._jit_tree_views.Expr) -> None"""

class Starred(Expr):
    def __init__(self, range: SourceRange, expr: Expr) -> None:
        """__init__(self: torch._C._jit_tree_views.Starred, arg0: torch._C._jit_tree_views.SourceRange, arg1: torch._C._jit_tree_views.Expr) -> None"""

class EmptyTypeAnnotation(TreeView):
    def __init__(self, range: SourceRange) -> None:
        """__init__(self: torch._C._jit_tree_views.EmptyTypeAnnotation, arg0: torch._C._jit_tree_views.SourceRange) -> None"""
