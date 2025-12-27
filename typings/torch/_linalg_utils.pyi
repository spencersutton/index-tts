"""Various linear algebra utility methods for internal use."""

from torch import Tensor

def is_sparse(A):
    """Check if tensor A is a sparse COO tensor. All other sparse storage formats (CSR, CSC, etc...) will return False."""

def get_floating_dtype(A):
    """
    Return the floating point dtype of tensor A.

    Integer types map to float32.
    """

def matmul(A: Tensor | None, B: Tensor) -> Tensor:
    """
    Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """

def bform(X: Tensor, A: Tensor | None, Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""

def qform(A: Tensor | None, S: Tensor):
    """Return quadratic form :math:`S^T A S`."""

def basis(A):
    """Return orthogonal basis of A columns."""

def symeig(A: Tensor, largest: bool | None = ...) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""

def matrix_rank(input, tol=..., symmetric=..., *, out=...) -> Tensor: ...
def solve(input: Tensor, A: Tensor, *, out=...) -> tuple[Tensor, Tensor]: ...
def lstsq(input: Tensor, A: Tensor, *, out=...) -> tuple[Tensor, Tensor]: ...
def eig(self: Tensor, eigenvectors: bool = ..., *, e=..., v=...) -> tuple[Tensor, Tensor]: ...
