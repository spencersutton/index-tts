from ._normalizations import (
    ArrayLike,
    ArrayLikeOrScalar,
    CastingModes,
    DTypeLike,
    NotImplementedType,
    OutArray,
    normalizer,
)

_binary = ...
NEP50_FUNCS = ...

def deco_binary_ufunc(torch_func):
    """
    Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: OutArray | None = ...,
    *,
    casting: CastingModes | None = ...,
    order: NotImplementedType = ...,
    dtype: DTypeLike | None = ...,
    subok: NotImplementedType = ...,
    signature: NotImplementedType = ...,
    extobj: NotImplementedType = ...,
    axes: NotImplementedType = ...,
    axis: NotImplementedType = ...,
): ...
@normalizer
def ldexp(
    x1: ArrayLikeOrScalar,
    x2: ArrayLikeOrScalar,
    /,
    out: OutArray | None = ...,
    *,
    where: NotImplementedType = ...,
    casting: CastingModes | None = ...,
    order: NotImplementedType = ...,
    dtype: DTypeLike | None = ...,
    subok: NotImplementedType = ...,
    signature: NotImplementedType = ...,
    extobj: NotImplementedType = ...,
): ...
@normalizer
def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: OutArray | None = ...,
    out2: OutArray | None = ...,
    /,
    out: tuple[OutArray | None, OutArray | None] = ...,
    *,
    where: NotImplementedType = ...,
    casting: CastingModes | None = ...,
    order: NotImplementedType = ...,
    dtype: DTypeLike | None = ...,
    subok: NotImplementedType = ...,
    signature: NotImplementedType = ...,
    extobj: NotImplementedType = ...,
): ...
def modf(x, /, *args, **kwds): ...

_binary = ...
_unary = ...
_fp_unary = ...

def deco_unary_ufunc(torch_func):
    """
    Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

__all__ = _binary + _unary
