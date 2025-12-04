from collections.abc import Iterable
from typing import Any, Tuple

from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils
from torchaudio._internal.module_utils import dropping_support

if _mod_utils.is_module_available("numpy"): ...
__all__ = ["read_mat_ark", "read_mat_scp", "read_vec_flt_ark", "read_vec_flt_scp", "read_vec_int_ark"]

@dropping_support
@_mod_utils.requires_module("kaldi_io", "numpy")
def read_vec_int_ark(file_or_fd: Any) -> Iterable[tuple[str, Tensor]]: ...
@dropping_support
@_mod_utils.requires_module("kaldi_io", "numpy")
def read_vec_flt_scp(file_or_fd: Any) -> Iterable[tuple[str, Tensor]]: ...
@dropping_support
@_mod_utils.requires_module("kaldi_io", "numpy")
def read_vec_flt_ark(file_or_fd: Any) -> Iterable[tuple[str, Tensor]]: ...
@dropping_support
@_mod_utils.requires_module("kaldi_io", "numpy")
def read_mat_scp(file_or_fd: Any) -> Iterable[tuple[str, Tensor]]: ...
@dropping_support
@_mod_utils.requires_module("kaldi_io", "numpy")
def read_mat_ark(file_or_fd: Any) -> Iterable[tuple[str, Tensor]]: ...
