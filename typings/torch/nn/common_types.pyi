from typing import TypeVar, TypeAlias
from torch import Tensor

T = TypeVar("T")
_size_any_t: TypeAlias = _scalar_or_tuple_any_t[int]
_size_1_t: TypeAlias = _scalar_or_tuple_1_t[int]
_size_2_t: TypeAlias = _scalar_or_tuple_2_t[int]
_size_3_t: TypeAlias = _scalar_or_tuple_3_t[int]
_size_4_t: TypeAlias = _scalar_or_tuple_4_t[int]
_size_5_t: TypeAlias = _scalar_or_tuple_5_t[int]
_size_6_t: TypeAlias = _scalar_or_tuple_6_t[int]
_size_any_opt_t: TypeAlias = _scalar_or_tuple_any_t[int | None]
_size_2_opt_t: TypeAlias = _scalar_or_tuple_2_t[int | None]
_size_3_opt_t: TypeAlias = _scalar_or_tuple_3_t[int | None]
_ratio_2_t: TypeAlias = _scalar_or_tuple_2_t[float]
_ratio_3_t: TypeAlias = _scalar_or_tuple_3_t[float]
_ratio_any_t: TypeAlias = _scalar_or_tuple_any_t[float]
_tensor_list_t: TypeAlias = _scalar_or_tuple_any_t[Tensor]
_maybe_indices_t: TypeAlias = _scalar_or_tuple_2_t[Tensor]
