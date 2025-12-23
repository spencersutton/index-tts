"""Tests for the verify_shapes decorator.

verify_shapes is intentionally non-fatal (prints warnings instead of raising),
so these tests focus on supported annotation forms.
"""

import sys
from pathlib import Path
from typing import Annotated

import torch
from torch import Tensor

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.util import verify_shapes


def test_verify_shapes_optional_tensor_none_is_ok(capsys) -> None:
    @verify_shapes
    def f(x: Annotated[Tensor, (2, 3)] | None) -> None:
        return None

    f(None)
    out = capsys.readouterr().out
    assert out == ""


def test_verify_shapes_optional_tensor_valid_tensor_is_checked(capsys) -> None:
    @verify_shapes
    def f(x: Annotated[Tensor, (2, 3)] | None) -> None:
        return None

    f(torch.zeros(2, 4))
    out = capsys.readouterr().out
    assert "Argument f:x has incorrect shape" in out


def test_verify_shapes_list_of_annotated_tensors_checks_each_element(capsys) -> None:
    @verify_shapes
    def f(xs: list[Annotated[Tensor, (2, 3)]]) -> None:
        return None

    good = torch.zeros(2, 3)
    bad = torch.zeros(2, 4)
    f([good, bad])

    out = capsys.readouterr().out
    assert "Argument f:xs[1] has incorrect shape" in out


def test_verify_shapes_optional_list_of_annotated_tensors_order_independent(capsys) -> None:
    # This ensures we support both (list[...] | None) and (None | list[...])
    @verify_shapes
    def f(xs: None | list[Annotated[Tensor, (2, 3)]]) -> None:
        return None

    f([torch.zeros(2, 4)])
    out = capsys.readouterr().out
    assert "Argument f:xs[0] has incorrect shape" in out
