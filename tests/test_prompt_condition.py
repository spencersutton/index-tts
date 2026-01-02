"""Fast unit tests for prompt-condition construction.

These tests avoid loading checkpoints / running full inference.
They verify that semantic-codec quantization for the prompt path is:
- skipped by default (for speed)
- used when explicitly enabled
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from indextts.infer_v2 import _build_prompt_condition


@dataclass
class DummyRegulator:
    last_x: torch.Tensor | None = None
    last_ylens: int | None = None

    def __call__(self, x: torch.Tensor, ylens: int) -> torch.Tensor:
        self.last_x = x
        self.last_ylens = ylens
        # Keep it simple: just return x unchanged.
        return x


@dataclass
class DummyCodec:
    delta: float = 1.0
    called: bool = False

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        self.called = True
        return x + self.delta


def test_build_prompt_condition_skips_quantize_by_default() -> None:
    x = torch.zeros(1, 3, 1024)
    reg = DummyRegulator()
    codec = DummyCodec(delta=1.0)

    out = _build_prompt_condition(
        spk_cond_emb=x,
        ref_mel_len=123,
        length_regulator=reg,
        semantic_codec=codec,
        use_semantic_codec=False,
    )

    assert codec.called is False
    assert reg.last_x is x
    assert reg.last_ylens == 123
    assert torch.equal(out, x)


def test_build_prompt_condition_uses_quantize_when_enabled() -> None:
    x = torch.zeros(1, 3, 1024)
    reg = DummyRegulator()
    codec = DummyCodec(delta=2.0)

    out = _build_prompt_condition(
        spk_cond_emb=x,
        ref_mel_len=7,
        length_regulator=reg,
        semantic_codec=codec,
        use_semantic_codec=True,
    )

    assert codec.called is True
    assert reg.last_ylens == 7
    assert reg.last_x is not None
    assert torch.equal(reg.last_x, x + 2.0)
    assert torch.equal(out, x + 2.0)


def test_build_prompt_condition_requires_codec_when_enabled() -> None:
    x = torch.zeros(1, 3, 1024)
    reg = DummyRegulator()

    with pytest.raises(ValueError, match="semantic_codec must be provided"):
        _build_prompt_condition(
            spk_cond_emb=x,
            ref_mel_len=5,
            length_regulator=reg,
            semantic_codec=None,
            use_semantic_codec=True,
        )
