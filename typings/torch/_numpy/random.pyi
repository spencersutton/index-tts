"""
Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy
"""

from ._normalizations import ArrayLike, normalizer

__all__ = [
    "choice",
    "normal",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "sample",
    "seed",
    "shuffle",
    "uniform",
]

def use_numpy_random(): ...
def deco_stream(func): ...
@deco_stream
def seed(seed=...): ...
@deco_stream
def random_sample(size=...): ...
def rand(*size): ...

sample = ...
random = ...

@deco_stream
def uniform(low=..., high=..., size=...): ...
@deco_stream
def randn(*size): ...
@deco_stream
def normal(loc=..., scale=..., size=...): ...
@deco_stream
def shuffle(x): ...
@deco_stream
def randint(low, high=..., size=...): ...
@deco_stream
@normalizer
def choice(a: ArrayLike, size=..., replace=..., p: ArrayLike | None = ...): ...
