import sys
import warnings
from pathlib import Path

import rich.traceback

from indextts.infer_v2 import IndexTTS2

if __debug__:
    import omegaconf
    import torch

    rich.traceback.install(suppress=[omegaconf, torch])

# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Profiling support (start/stop only around inference)
profiler = None


def main() -> None:
    import argparse

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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling (saves to indextts.json, requires viztracer)",
    )
    args = parser.parse_args()

    assert isinstance(args.text, str)  # pyright: ignore[reportAny]
    assert isinstance(args.voice, str)  # pyright: ignore[reportAny]
    assert isinstance(args.config, str)  # pyright: ignore[reportAny]
    assert isinstance(args.output_path, str)  # pyright: ignore[reportAny]
    assert isinstance(args.model_dir, str)  # pyright: ignore[reportAny]
    assert isinstance(args.device, (str, type(None)))  # pyright: ignore[reportAny]
    assert isinstance(args.fp16, bool)  # pyright: ignore[reportAny]
    assert isinstance(args.force, bool)  # pyright: ignore[reportAny]
    assert isinstance(args.use_accel, bool)  # pyright: ignore[reportAny]
    assert isinstance(args.use_torch_compile, bool)  # pyright: ignore[reportAny]
    assert isinstance(args.use_cuda_kernel, bool)  # pyright: ignore[reportAny]
    assert isinstance(args.use_deepspeed, bool)  # pyright: ignore[reportAny]

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

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        sys.exit(1)

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
    # If profiling was requested, start profiler immediately before inference
    if args.profile:
        try:
            from viztracer import VizTracer

            # Increase buffer size to 10M entries to prevent wrapping on long runs
            # ignore_c_function=True reduces noise from builtins, but you can remove it if you need to see C calls
            profiler = VizTracer(tracer_entries=10000000, ignore_c_function=True)
            profiler.start()
        except ImportError:
            print("PROFILING ERROR: 'viztracer' is not installed.")
            print("Please install it to use --profile: pip install viztracer")
            sys.exit(1)

    # Run inference and ensure profiling only captures this call
    try:
        tts.infer(
            spk_audio_prompt=args.voice,
            text=args.text.strip(),
            output_path=output_path,
        )
    finally:
        # Stop and save profiler immediately after inference so we only capture inference time
        if args.profile and profiler:
            try:
                profiler.stop()
                profiler.save("indextts.json")
                print("\nProfiling data saved to indextts.json")
                print("Visualization options:")
                print("1. Perfetto (Recommended): Open https://ui.perfetto.dev/ and load indextts.json")
                print("2. Chrome Tracing: Open chrome://tracing and load indextts.json")
                print("3. VizTracer: vizviewer indextts.json")
            finally:
                # Clear profiler so module-level finally will not attempt to stop it again
                profiler = None


if __name__ == "__main__":
    try:
        main()
    finally:
        if profiler:
            profiler.stop()
            profiler.save("indextts.json")
            print("\nProfiling data saved to indextts.json")
            print("Visualization options:")
            print("1. Perfetto (Recommended): Open https://ui.perfetto.dev/ and load indextts.json")
            print("2. Chrome Tracing: Open chrome://tracing and load indextts.json")
            print("3. VizTracer: vizviewer indextts.json")
