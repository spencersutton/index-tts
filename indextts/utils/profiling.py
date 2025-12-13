#!/usr/bin/env python3
"""Configuration helpers for filtered torch profiling.

This module provides utilities to profile with torch.profiler while
keeping output manageable and focused on your code.
"""

from pathlib import Path
from typing import Callable

import torch


def create_filtered_profiler(
    output_path: str | Path = "profile_trace.json",
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = True,
    with_modules: bool = True,
):
    """Create a torch profiler with smart filtering.

    Args:
        output_path: Where to save the trace (Chrome trace format)
        record_shapes: Record tensor shapes (increases file size)
        profile_memory: Profile memory usage (increases file size)
        with_stack: Include Python stack traces
        with_modules: Include module hierarchy

    Returns:
        Configured torch.profiler.profile context manager

    Example:
        with create_filtered_profiler("my_profile.json") as prof:
            model(input)
        # View with: chrome://tracing or https://ui.perfetto.dev/
    """

    def trace_handler(prof):
        """Export trace after profiling."""
        print(f"\nExporting trace to {output_path}...")
        prof.export_chrome_trace(str(output_path))
        print("✓ Trace saved. View at chrome://tracing or https://ui.perfetto.dev/")

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_modules=with_modules,
        # Use schedule to limit recording duration for long runs
        schedule=torch.profiler.schedule(
            wait=0,  # Don't wait before profiling
            warmup=1,  # Warmup for 1 step
            active=3,  # Profile for 3 steps
            repeat=1,  # Only do this once
        ),
        on_trace_ready=trace_handler,
    )


def print_filtered_table(
    prof: torch.profiler.profile,
    sort_by: str = "cuda_time_total",
    row_limit: int = 30,
    max_name_column_width: int = 80,
):
    """Print profiler table with filtering for readability.

    Args:
        prof: Profiler instance after profiling
        sort_by: Sort key (cuda_time_total, cpu_time_total, cpu_time, etc.)
        row_limit: Max rows to display
        max_name_column_width: Max width for operation name column

    Example:
        with torch.profiler.profile(...) as prof:
            model(input)
        print_filtered_table(prof, sort_by="cuda_time_total", row_limit=50)
    """

    # Get table as string
    table = prof.key_averages().table(
        sort_by=sort_by,
        row_limit=row_limit,
        max_name_column_width=max_name_column_width,
    )

    print("\n" + "=" * 100)
    print(f"TORCH PROFILER RESULTS (Top {row_limit} by {sort_by})")
    print("=" * 100)
    print(table)
    print("=" * 100)


def filter_by_module(
    prof: torch.profiler.profile,
    module_prefixes: list[str] = None,
    exclude_prefixes: list[str] = None,
):
    """Filter profiler events by module name.

    Args:
        prof: Profiler instance
        module_prefixes: Only include modules starting with these (e.g., ["indextts", "torch.nn"])
        exclude_prefixes: Exclude modules starting with these (e.g., ["torch._", "torch.autograd"])

    Returns:
        Filtered event list

    Example:
        with torch.profiler.profile(with_modules=True) as prof:
            model(input)

        # Show only your code and torch.nn
        events = filter_by_module(prof, module_prefixes=["indextts", "torch.nn"])
        for evt in events[:20]:
            print(f"{evt.key}: {evt.cuda_time_total/1000:.2f}ms")
    """
    if module_prefixes is None:
        module_prefixes = ["indextts"]

    if exclude_prefixes is None:
        exclude_prefixes = ["torch._", "torch.autograd", "torch.cuda"]

    all_events = prof.key_averages()
    filtered = []

    for evt in all_events:
        key = evt.key

        # Check exclusions first
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        # Check inclusions
        if any(key.startswith(prefix) for prefix in module_prefixes):
            filtered.append(evt)

    return filtered


def profile_function_calls(
    func: Callable,
    *args,
    trace_path: str | Path = "profile_trace.json",
    print_table: bool = True,
    table_rows: int = 30,
    **kwargs,
):
    """Profile a function call with sensible defaults.

    Args:
        func: Function to profile
        *args: Arguments to pass to func
        trace_path: Where to save trace file
        print_table: Whether to print summary table
        table_rows: Number of rows in summary table
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)

    Example:
        result = profile_function_calls(
            tts.infer,
            spk_audio_prompt="voice.wav",
            text="Hello",
            output_path="out.wav",
            trace_path="inference_profile.json",
        )
    """

    with create_filtered_profiler(trace_path) as prof:
        result = func(*args, **kwargs)

    if print_table:
        print_filtered_table(prof, row_limit=table_rows)

    return result


# Example usage in a script:
if __name__ == "__main__":
    print(__doc__)
    print("\nExample usage:\n")
    print("1. Basic profiling:")
    print("   from indextts.utils.profiling import create_filtered_profiler")
    print("   with create_filtered_profiler('my_trace.json') as prof:")
    print("       tts.infer(...)")
    print()
    print("2. Quick function profiling:")
    print("   from indextts.utils.profiling import profile_function_calls")
    print("   profile_function_calls(tts.infer, spk_audio_prompt='voice.wav', text='Hello')")
    print()
    print("3. View traces at: chrome://tracing or https://ui.perfetto.dev/")
