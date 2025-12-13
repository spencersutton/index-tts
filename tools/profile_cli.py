#!/usr/bin/env python3
"""Profile cli.py with filtered output showing only relevant code.

This profiles your code and direct library calls without deep library traces.

Usage:
    python tools/profile_cli.py -v <voice_file> "text to synthesize"
    
    # With options
    python tools/profile_cli.py -v outputs/mizora.ogg "Hello world" --use-accel
    
    # Profile top 50 functions
    python tools/profile_cli.py -v outputs/mizora.ogg "test" --top 50
"""

import argparse
import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path


def should_include_function(filename: str, project_root: str) -> bool:
    """Determine if a function should be included in the profile output.
    
    Includes:
    - Your project code (indextts/)
    - Top-level calls from key libraries (torch, transformers, etc.)
    
    Excludes:
    - Deep library internals
    - Python stdlib (unless direct calls)
    """
    # Always include your own code
    if project_root in filename or "indextts" in filename:
        return True
    
    # Include top-level library calls but filter out deep internals
    # You can adjust these patterns based on what you want to see
    include_patterns = [
        "torch/nn/",  # Top-level torch.nn operations
        "transformers/models/",  # Model architectures
        "transformers/generation/",  # Generation logic
    ]
    
    # Exclude deep library internals
    exclude_patterns = [
        "site-packages/torch/_",  # Torch internals
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
    
    # Check exclusions first (they take precedence)
    for pattern in exclude_patterns:
        if pattern in filename:
            return False
    
    # Check inclusions
    for pattern in include_patterns:
        if pattern in filename:
            return True
    
    # By default, exclude library code unless explicitly included
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
        description="Profile IndexTTS CLI with filtered output"
    )
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument(
        "-v", "--voice", required=True, help="Path to voice prompt audio file"
    )
    parser.add_argument(
        "-o", "--output-path", default="gen.wav", help="Output path"
    )
    parser.add_argument(
        "--config", default="checkpoints/config.yaml", help="Config file path"
    )
    parser.add_argument("--model-dir", default="checkpoints", help="Model directory")
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
        help="Number of top functions to show (default: 30)"
    )
    parser.add_argument(
        "--save-full-profile",
        help="Save complete unfiltered profile to this file (e.g., profile.pstats)"
    )
    
    args = parser.parse_args()
    
    # Setup for running the inference
    project_root = str(Path(__file__).parent.parent.absolute())
    sys.path.insert(0, project_root)
    
    import torch
    from indextts.infer_v2 import IndexTTS2
    
    # Auto-detect device
    device = args.device
    fp16 = args.fp16
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            fp16 = False
    
    print(f"Initializing IndexTTS on {device}...")
    tts = IndexTTS2(
        cfg_path=Path(args.config),
        model_dir=Path(args.model_dir),
        use_fp16=fp16,
        device=device,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
    )
    
    print(f"\nRunning inference with profiling...")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}\n")
    
    # Profile the inference
    profiler = cProfile.Profile()
    profiler.enable()
    
    tts.infer(
        spk_audio_prompt=Path(args.voice),
        text=args.text.strip(),
        output_path=Path(args.output_path),
    )
    
    profiler.disable()
    
    # Save full profile if requested
    if args.save_full_profile:
        profiler.dump_stats(args.save_full_profile)
        print(f"\nFull profile saved to: {args.save_full_profile}")
    
    # Print filtered stats
    stats = pstats.Stats(profiler)
    print_filtered_stats(stats, project_root, args.top)
    
    print(f"\n✓ Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
