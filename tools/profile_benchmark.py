#!/usr/bin/env python3
"""Profile benchmark.py runs with filtered output.

This profiles benchmark runs showing only your code and top-level library calls.

Usage:
    python tools/profile_benchmark.py -v <voice_file> [options]
    
Examples:
    # Basic profiling with 3 runs
    python tools/profile_benchmark.py -v outputs/mizora.ogg --runs 3
    
    # With torch.compile and detailed output
    python tools/profile_benchmark.py -v outputs/mizora.ogg --use_torch_compile --top 50
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path


def should_include_function(filename: str, project_root: str) -> bool:
    """Determine if a function should be included in the profile output."""
    # Always include your own code
    if project_root in filename or "indextts" in filename or "benchmark" in filename:
        return True
    
    # Include top-level library calls
    include_patterns = [
        "torch/nn/",
        "transformers/models/",
        "transformers/generation/",
    ]
    
    # Exclude deep library internals
    exclude_patterns = [
        "site-packages/torch/_",
        "site-packages/torch/utils/",
        "/torch/jit/",
        "/torch/autograd/",
        "/torch/cuda/",
        "site-packages/transformers/_",
        "/typing.py",
        "/abc.py",
        "/contextlib.py",
        "/functools.py",
        "/copy.py",
        "/warnings.py",
        "/<frozen",
        "<built-in",
    ]
    
    # Check exclusions first
    for pattern in exclude_patterns:
        if pattern in filename:
            return False
    
    # Check inclusions
    for pattern in include_patterns:
        if pattern in filename:
            return True
    
    # By default, exclude library code
    if "site-packages" in filename or "/lib/python" in filename:
        return False
    
    return True


def print_filtered_stats(stats: pstats.Stats, project_root: str, top_n: int = 30):
    """Print profile stats filtered to show only relevant functions."""
    
    # Get the raw stats
    stats_data = stats.stats
    
    # Filter the stats
    filtered_stats = {}
    for func_key, func_stats in stats_data.items():
        filename, line, func_name = func_key
        if should_include_function(filename, project_root):
            filtered_stats[func_key] = func_stats
    
    # Create a new Stats object with filtered data
    filtered = pstats.Stats()
    filtered.stats = filtered_stats
    
    # Calculate totals
    filtered.total_calls = sum(cc for cc, nc, tt, ct, callers in filtered_stats.values())
    filtered.prim_calls = sum(nc for cc, nc, tt, ct, callers in filtered_stats.values())
    filtered.total_tt = sum(tt for cc, nc, tt, ct, callers in filtered_stats.values())
    
    print("\n" + "=" * 100)
    print("FILTERED PROFILE RESULTS (Your Code + Top-Level Library Calls)")
    print("=" * 100)
    print(f"Total filtered calls: {filtered.total_calls}")
    print(f"Total time in filtered functions: {filtered.total_tt:.3f}s")
    print("\nNote: Times include all subfunctions (even if filtered out from display)\n")
    
    # Print stats sorted by cumulative time
    print("-" * 100)
    print("Top functions by CUMULATIVE time (includes time in called functions):")
    print("-" * 100)
    filtered.sort_stats('cumulative')
    filtered.print_stats(top_n)
    
    print("\n" + "-" * 100)
    print("Top functions by INTERNAL time (time spent in function itself):")
    print("-" * 100)
    filtered.sort_stats('time')
    filtered.print_stats(top_n)


def main():
    parser = argparse.ArgumentParser(
        description="Profile IndexTTS benchmark with filtered output"
    )
    parser.add_argument(
        "-v", "--voice", required=True, help="Path to voice prompt audio file"
    )
    parser.add_argument(
        "--text",
        default="This is a benchmark test for IndexTTS inference performance.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--config", default="checkpoints/config.yaml", help="Config file path"
    )
    parser.add_argument("--model-dir", default="checkpoints", help="Model directory")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--device", default=None, help="Device (cuda, cpu, etc)")
    parser.add_argument("--use-accel", action="store_true", help="Use acceleration")
    parser.add_argument("--use-torch-compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--use-cuda-kernel", action="store_true", help="Use CUDA kernel")
    parser.add_argument("--use-deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top functions to show (default: 30)",
    )
    parser.add_argument(
        "--save-full-profile",
        help="Save complete unfiltered profile to this file (e.g., profile.pstats)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = str(Path(__file__).parent.parent.absolute())
    sys.path.insert(0, project_root)
    
    # Import benchmark module
    from tools.benchmark import run_benchmark
    
    print(f"Running benchmark with profiling...")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Runs: {args.runs} (+ {args.warmup} warmup)\n")
    
    # Profile the benchmark
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = run_benchmark(
        voice_path=args.voice,
        text=args.text,
        config=Path(args.config),
        model_dir=Path(args.model_dir),
        num_runs=args.runs,
        warmup_runs=args.warmup,
        fp16=args.fp16,
        device=args.device,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        verbose=True,
    )
    
    profiler.disable()
    
    # Print benchmark results
    result.print_report()
    
    # Save full profile if requested
    if args.save_full_profile:
        profiler.dump_stats(args.save_full_profile)
        print(f"\nFull profile saved to: {args.save_full_profile}")
    
    # Print filtered stats
    stats = pstats.Stats(profiler)
    stats.dump_stats("full_profile.pstats")
    print_filtered_stats(stats, project_root, args.top)


if __name__ == "__main__":
    main()
