#!/usr/bin/env python3
"""Quick comparison of profiling approaches.

This script demonstrates the difference between unfiltered and filtered profiling.
"""

import tempfile
from pathlib import Path

print("=" * 80)
print("PROFILING TOOLS COMPARISON")
print("=" * 80)

print("""
You have 3 main options for profiling without huge traces:

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Python cProfile with Filtering (Recommended for most cases)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Fast and lightweight                                                      │
│ • Shows YOUR code + top-level library calls                                 │
│ • Text output, easy to read                                                 │
│ • Works on CPU and GPU                                                      │
│                                                                              │
│ Usage:                                                                      │
│   python tools/profile_cli.py -v voice.wav "test"                          │
│   python tools/profile_benchmark.py -v voice.wav --runs 3                  │
│                                                                              │
│ Output: ~50-200 lines of readable text showing where time is spent         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Torch Profiler with Smart Config (For GPU kernel analysis)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ • GPU/CUDA kernel details                                                   │
│ • Visual timeline in Chrome                                                 │
│ • Automatically limits recording to 3-4 steps                               │
│ • Manageable file size (5-50 MB instead of 500+ MB)                         │
│                                                                              │
│ Usage:                                                                      │
│   python cli_with_profiling.py -v voice.wav "test" --profile               │
│                                                                              │
│ Output: JSON trace file viewable in chrome://tracing                       │
│ Size: ~10-50 MB (vs 500+ MB unfiltered)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Custom Profiling in Your Code                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Import from indextts.utils.profiling                                      │
│ • Full control over what/when to profile                                    │
│ • Combine with your own timing code                                         │
│                                                                              │
│ Usage:                                                                      │
│   from indextts.utils.profiling import profile_function_calls             │
│   profile_function_calls(my_func, arg1, arg2, trace_path="out.json")      │
└─────────────────────────────────────────────────────────────────────────────┘

""")

print("COMPARISON OF FILE SIZES:")
print("-" * 80)
print("Approach                              Typical Size    Detail Level")
print("-" * 80)
print("Original cli.py (torch profiler)      500+ MB        Everything (too much)")
print("profile_cli.py (filtered cProfile)    N/A (text)     Your code + top libs")
print("cli_with_profiling.py (smart torch)   10-50 MB       GPU details, limited steps")
print("profile_benchmark.py                  N/A (text)     Multiple runs averaged")
print("-" * 80)

print("\nWHAT YOU'LL SEE IN FILTERED OUTPUT:")
print("-" * 80)
print("✓ indextts.infer_v2.infer()          - Your main inference function")
print("✓ indextts.gpt.model_v2.forward()    - Your model forward passes")
print("✓ torch.nn.Linear.forward()          - Top-level PyTorch operations")
print("✓ transformers.generation.generate() - Generation logic")
print()
print("✗ torch._C._cuda_getCurrentRawStream() - Deep CUDA internals (filtered)")
print("✗ torch.autograd.Function.apply()      - Autograd internals (filtered)")
print("✗ typing.get_type_hints()              - Python typing (filtered)")
print("-" * 80)

print("\nQUICK START:")
print("-" * 80)
print("1. For quick profiling:     python tools/profile_cli.py -v voice.wav 'text'")
print("2. For benchmark profiling: python tools/profile_benchmark.py -v voice.wav")
print("3. For GPU kernel details:  python cli_with_profiling.py -v voice.wav 'text' --profile")
print("-" * 80)

print("\nFor detailed documentation, see: tools/PROFILING.md")
print("=" * 80)
