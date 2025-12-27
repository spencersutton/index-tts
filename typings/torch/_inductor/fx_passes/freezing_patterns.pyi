import functools

import torch

from ..pattern_matcher import CallFunction, Ignored, KeywordArg, Match, init_once_fakemode, register_graph_pattern

aten = ...
pass_patterns = ...
binary_folding_pass = ...

def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """Passes that are applied to the graph to freeze pass."""

@init_once_fakemode
def lazy_init(): ...
def register_freezing_graph_pattern(pattern, extra_check=..., pass_number=...): ...
def register_binary_folding_pattern(pattern, extra_check=...): ...
@functools.cache
def addmm_patterns_init():
    """
    addmm related patterns.
    To avoid duplication, also includes int8 WoQ GEMM pattern without bias.
    """

def same_dtype(match): ...
@register_graph_pattern(
    CallFunction(torch.ops.prims.convert_element_type.default, Ignored(), KeywordArg("dtype")),
    pass_dict=pass_patterns[0],
    extra_check=same_dtype,
)
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
