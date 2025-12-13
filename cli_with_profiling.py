#!/usr/bin/env python3
"""CLI with optional smart profiling.

Run with profiling:
    python cli_with_profiling.py -v voice.wav "text" --profile

Run with profiling and custom trace output:
    python cli_with_profiling.py -v voice.wav "text" --profile --profile-output my_trace.json

Run without profiling (normal mode):
    python cli_with_profiling.py -v voice.wav "text"
"""

import argparse
import sys
import warnings
from pathlib import Path

import rich.traceback
import torch

from indextts.infer_v2 import IndexTTS2

if __debug__:
    import omegaconf
    import transformers

    rich.traceback.install(suppress=[omegaconf, torch, transformers], width=120)

# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser(description="IndexTTS Command Line")
parser.add_argument("text", type=str, help="Text to be synthesized")
parser.add_argument(
    "-v",
    "--voice",
    type=str,
    required=True,
    help="Path to the audio prompt file (wav format)",
)
parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    default="gen.wav",
    help="Path to the output wav file",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="checkpoints/config.yaml",
    help="Path to the config file. Default is 'checkpoints/config.yaml'",
)
parser.add_argument(
    "--model-dir",
    type=str,
    default="checkpoints",
    help="Path to the model directory. Default is 'checkpoints'",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    default=False,
    help="Use FP16 for inference if available",
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    default=False,
    help="Force to overwrite the output file if it exists",
)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default=None,
    help="Device to run the model on (cpu, cuda, mps, xpu).",
)
parser.add_argument(
    "--use-accel",
    action="store_true",
    default=False,
    help="Use acceleration engine (FlashAttention) for GPT",
)
parser.add_argument(
    "--use-torch-compile",
    action="store_true",
    default=False,
    help="Use torch.compile for optimization",
)
parser.add_argument(
    "--use-cuda-kernel",
    action="store_true",
    default=False,
    help="Use custom CUDA kernel for BigVGAN",
)
parser.add_argument(
    "--use-deepspeed",
    action="store_true",
    default=False,
    help="Use DeepSpeed for inference",
)

# Profiling options
parser.add_argument(
    "--profile",
    action="store_true",
    default=False,
    help="Enable profiling (creates trace file and prints summary)",
)
parser.add_argument(
    "--profile-output",
    type=str,
    default="profile_trace.json",
    help="Path to save profiling trace (default: profile_trace.json)",
)
parser.add_argument(
    "--profile-rows",
    type=int,
    default=30,
    help="Number of rows to show in profile summary (default: 30)",
)
parser.add_argument(
    "--profile-memory",
    action="store_true",
    default=False,
    help="Include memory profiling (increases trace size)",
)

args = parser.parse_args()

assert isinstance(args.text, str)
assert isinstance(args.voice, str)
assert isinstance(args.config, str)
assert isinstance(args.output_path, str)
assert isinstance(args.model_dir, str)
assert isinstance(args.device, (str, type(None)))
assert isinstance(args.fp16, bool)
assert isinstance(args.force, bool)
assert isinstance(args.use_accel, bool)
assert isinstance(args.use_torch_compile, bool)
assert isinstance(args.use_cuda_kernel, bool)
assert isinstance(args.use_deepspeed, bool)

if len(args.text.strip()) == 0:
    print("ERROR: Text is empty.")
    parser.print_help()
    sys.exit(1)
if not Path(args.voice).exists():
    print(f"Audio prompt file {args.voice} does not exist.")
    parser.print_help()
    sys.exit(1)
if not Path(args.config).exists():
    print(f"Config file {args.config} does not exist.")
    parser.print_help()
    sys.exit(1)

output_path = Path(args.output_path)
if output_path.exists():
    if not args.force:
        print(f"ERROR: Output file {output_path} already exists. Use --force to overwrite.")
        parser.print_help()
        sys.exit(1)
    else:
        output_path.unlink()

if args.device is None:
    if torch.cuda.is_available():
        args.device = "cuda:0"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        args.device = "xpu"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
        args.fp16 = False  # Disable FP16 on CPU
        print("WARNING: Running on CPU may be slow.")

tts = IndexTTS2(
    cfg_path=Path(args.config),
    model_dir=Path(args.model_dir),
    use_fp16=args.fp16,
    device=args.device,
    use_accel=args.use_accel,
    use_torch_compile=args.use_torch_compile,
    use_cuda_kernel=args.use_cuda_kernel,
    use_deepspeed=args.use_deepspeed,
)

if args.profile:
    # Use smart profiling
    from indextts.utils.profiling import create_filtered_profiler, print_filtered_table

    print(f"\nProfiling enabled. Trace will be saved to: {args.profile_output}")
    print("View trace at: chrome://tracing or https://ui.perfetto.dev/\n")

    with create_filtered_profiler(
        output_path=args.profile_output,
        profile_memory=args.profile_memory,
    ) as prof:
        tts.infer(
            spk_audio_prompt=Path(args.voice),
            text=args.text.strip(),
            output_path=output_path,
        )
        prof.step()  # Trigger trace save

    # Print summary table
    print_filtered_table(prof, row_limit=args.profile_rows)

else:
    # Normal execution without profiling
    tts.infer(
        spk_audio_prompt=Path(args.voice),
        text=args.text.strip(),
        output_path=output_path,
    )

print(f"\n✓ Audio generated successfully: {output_path}")
