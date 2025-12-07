#!/usr/bin/env python3
"""
Benchmark script for IndexTTS inference performance.

This script separates startup time from inference time to allow accurate
performance measurement of repeated inference runs.

Usage:
    python benchmark.py -v <voice_file> [options]

Examples:
    # Basic benchmark with 5 runs
    python benchmark.py -v outputs/mizora.ogg

    # Benchmark with custom text and warmup
    python benchmark.py -v outputs/mizora.ogg --text "Hello world" --warmup 3 --runs 20

    # Benchmark with torch.compile optimization
    python benchmark.py -v outputs/mizora.ogg --use_torch_compile
"""

import argparse
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _get_audio_duration(path: Path) -> float:
    """Return the duration in seconds of an audio file.

    Tries soundfile (pysoundfile) first, then falls back to wave for WAV files.
    Returns 0.0 on failure.
    """
    if not path or not Path(path).exists():
        return 0.0

    # try pysoundfile if available (supports many formats)
    try:
        import soundfile as sf

        info = sf.info(path)
        if hasattr(info, "duration") and info.duration is not None:
            return float(info.duration)
        # fallback: read frames and compute
        data, sr = sf.read(path, always_2d=False)
        frames = data.shape[0] if hasattr(data, "shape") else len(data)
        return frames / float(sr) if sr and sr > 0 else 0.0
    except Exception:
        pass

    # fallback for wav files using stdlib wave module
    try:
        import wave

        if path.suffix == ".wav":
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate) if rate and rate > 0 else 0.0
    except Exception:
        pass

    # can't determine duration
    print(f"Warning: unable to determine audio duration for {path}")
    return 0.0


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    startup_time: float
    text: str
    num_runs: int
    warmup_times: list[float] = field(default_factory=list)
    inference_times: list[float] = field(default_factory=list)
    # durations (seconds) of output audio produced by each run
    warmup_durations: list[float] = field(default_factory=list)
    inference_durations: list[float] = field(default_factory=list)

    @property
    def mean_inference(self) -> float:
        return statistics.mean(self.inference_times)

    @property
    def mean_duration(self) -> float:
        """Mean duration of the generated audio over the benchmark runs.

        Note: only includes durations where a positive value was recorded.
        """
        vals = [d for d in self.inference_durations if d and d > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def median_duration(self) -> float:
        vals = [d for d in self.inference_durations if d and d > 0]
        return statistics.median(vals) if vals else 0.0

    @property
    def std_inference(self) -> float:
        return statistics.stdev(self.inference_times) if len(self.inference_times) > 1 else 0.0

    @property
    def min_inference(self) -> float:
        return min(self.inference_times)

    @property
    def max_inference(self) -> float:
        return max(self.inference_times)

    @property
    def median_inference(self) -> float:
        return statistics.median(self.inference_times)

    # RTF: real-time factor = inference_time / output_audio_duration
    @property
    def rtf_list(self) -> list[float]:
        pairs = []
        for t, d in zip(self.inference_times, self.inference_durations):
            try:
                if d and d > 0:
                    pairs.append(t / d)
            except Exception:
                continue
        return pairs

    @property
    def mean_rtf(self) -> float:
        vals = self.rtf_list
        return statistics.mean(vals) if vals else 0.0

    @property
    def std_rtf(self) -> float:
        vals = self.rtf_list
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    @property
    def min_rtf(self) -> float:
        vals = self.rtf_list
        return min(vals) if vals else 0.0

    @property
    def max_rtf(self) -> float:
        vals = self.rtf_list
        return max(vals) if vals else 0.0

    @property
    def median_rtf(self) -> float:
        vals = self.rtf_list
        return statistics.median(vals) if vals else 0.0

    def print_report(self) -> None:
        """Print a formatted benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print(f'\nText: "{self.text}"')
        print(f"Text length: {len(self.text)} characters")

        print("\n--- Startup Time ---")
        print(f"  Model loading: {self.startup_time:.3f}s")

        if self.warmup_times:
            print(f"\n--- Warmup Runs ({len(self.warmup_times)} runs) ---")
            for i, t in enumerate(self.warmup_times, 1):
                print(f"  Warmup {i}: {t:.3f}s")

        print(f"\n--- Inference Performance ({self.num_runs} runs) ---")
        print(f"  Mean:     {self.mean_inference:.3f}s")
        print(f"  Std Dev:  {self.std_inference:.3f}s")
        print(f"  Min:      {self.min_inference:.3f}s")
        print(f"  Max:      {self.max_inference:.3f}s")
        print(f"  Median:   {self.median_inference:.3f}s")

        # durations
        if any(d and d > 0 for d in self.inference_durations):
            print("\n--- Output Durations (per-run) ---")
            for i, d in enumerate(self.inference_durations, 1):
                print(f"  Run {i:3d}: {d:.3f}s")

        # RTF stats
        if self.rtf_list:
            print("\n--- Real-Time Factor (RTF) ---")
            print(f"  Mean RTF:   {self.mean_rtf:.3f}")
            print(f"  Std RTF:    {self.std_rtf:.3f}")
            print(f"  Min RTF:    {self.min_rtf:.3f}")
            print(f"  Max RTF:    {self.max_rtf:.3f}")
            print(f"  Median RTF: {self.median_rtf:.3f}")

        print("\n--- Individual Run Times ---")
        for i, t in enumerate(self.inference_times, 1):
            print(f"  Run {i:3d}: {t:.3f}s")

        print("\n" + "=" * 60)

        # Summary comparison
        print("\nSUMMARY:")
        print(f"  Startup overhead:      {self.startup_time:.3f}s (one-time cost)")
        print(f"  Per-inference time:    {self.mean_inference:.3f}s ± {self.std_inference:.3f}s")
        if self.mean_duration:
            print(f"  Mean output duration:  {self.mean_duration:.3f}s")
        if self.mean_rtf:
            print(f"  Mean RTF:              {self.mean_rtf:.3f} ± {self.std_rtf:.3f}")
        if self.warmup_times:
            first_warmup = self.warmup_times[0]
            print(f"  First run penalty:     {first_warmup - self.mean_inference:+.3f}s vs mean")
        print("=" * 60)


def run_benchmark(
    voice_path: str,
    text: str = "This is a benchmark test for IndexTTS inference performance.",
    config: Path = Path("checkpoints/config.yaml"),
    model_dir: Path = Path("checkpoints"),
    num_runs: int = 5,
    warmup_runs: int = 1,
    output_path: Path | None = None,
    fp16: bool = False,
    device: str | None = None,
    use_accel: bool = False,
    use_torch_compile: bool = False,
    use_cuda_kernel: bool = False,
    use_deepspeed: bool = False,
    verbose: bool = False,
) -> BenchmarkResult:
    """
    Run the benchmark and return results.

    Args:
        voice_path: Path to the voice prompt audio file
        text: Text to synthesize
        config: Path to config file
        model_dir: Path to model directory
        num_runs: Number of inference runs to benchmark
        warmup_runs: Number of warmup runs before benchmarking
        output_path: Path to save output (optional, uses temp file if None)
        fp16: Use FP16 inference
        device: Device to use (auto-detected if None)
        use_accel: Use acceleration engine
        use_torch_compile: Use torch.compile optimization
        use_cuda_kernel: Use custom CUDA kernel for BigVGAN
        use_deepspeed: Use DeepSpeed for inference
        verbose: Print verbose output

    Returns:
        BenchmarkResult with timing data
    """
    import tempfile

    import torch

    from indextts.infer_v2 import IndexTTS2

    # Auto-detect device if not specified
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
            print("WARNING: Running on CPU may be slow.")

    print(f"\n{'=' * 60}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Voice prompt:      {voice_path}")
    print(f'  Text:              "{text[:50]}{"..." if len(text) > 50 else ""}"')
    print(f"  Device:            {device}")
    print(f"  FP16:              {fp16}")
    print(f"  Warmup runs:       {warmup_runs}")
    print(f"  Benchmark runs:    {num_runs}")
    print(f"  use-accel:         {use_accel}")
    print(f"  use-torch-compile: {use_torch_compile}")
    print(f"  use-cuda-kernel:   {use_cuda_kernel}")
    print(f"  use-deepspeed:     {use_deepspeed}")
    print(f"{'=' * 60}\n")

    # Determine output path

    if output_path is None:
        temp_dir = tempfile.mkdtemp(prefix="indextts_benchmark_")
        output_path = Path(temp_dir) / "benchmark_output.wav"

    # Measure startup time
    print("Loading model...")
    startup_start = time.perf_counter()

    tts = IndexTTS2(
        cfg_path=config,
        model_dir=model_dir,
        use_fp16=fp16,
        device=device,
        use_accel=use_accel,
        use_torch_compile=use_torch_compile,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
    )

    startup_time = time.perf_counter() - startup_start
    print(f"Model loaded in {startup_time:.3f}s\n")

    # Warmup runs
    warmup_times = []
    warmup_durations = []
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup inference(s)...")
        for i in range(warmup_runs):
            start = time.perf_counter()
            tts.infer(
                spk_audio_prompt=voice_path,
                text=text,
                output_path=output_path,
                verbose=verbose,
            )
            elapsed = time.perf_counter() - start
            warmup_times.append(elapsed)
            # record produced output audio duration
            warmup_durations.append(_get_audio_duration(output_path))
            print(f"  Warmup {i + 1}: {elapsed:.3f}s")
        print()

    # Benchmark runs
    inference_times = []
    inference_durations = []
    print(f"Running {num_runs} benchmark inference(s)...")
    for i in range(num_runs):
        start = time.perf_counter()
        tts.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=output_path,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)
        # read audio duration from the generated file
        inference_durations.append(_get_audio_duration(output_path))
        print(f"  Run {i + 1}/{num_runs}: {elapsed:.3f}s")

    return BenchmarkResult(
        startup_time=startup_time,
        warmup_times=warmup_times,
        inference_times=inference_times,
        warmup_durations=warmup_durations,
        inference_durations=inference_durations,
        text=text,
        num_runs=num_runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark IndexTTS inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py -v outputs/mizora.ogg
  python benchmark.py -v outputs/mizora.ogg --runs 20 --warmup 3
  python benchmark.py -v outputs/mizora.ogg --use_torch_compile --text "Custom text"
        """,
    )

    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        required=True,
        help="Path to the audio prompt file (wav/ogg format)",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="This is a benchmark test for IndexTTS inference performance.",
        help="Text to synthesize for benchmarking",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Path to the model directory",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before benchmarking (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save output audio (uses temp file if not specified)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 for inference if available",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to run on (cpu, cuda, mps, xpu). Auto-detected if not specified.",
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
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose output during inference",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to append results to a CSV file",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.voice).exists():
        print(f"ERROR: Voice prompt file {args.voice} does not exist.")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"ERROR: Config file {args.config} does not exist.")
        sys.exit(1)

    if args.runs < 1:
        print("ERROR: Number of runs must be at least 1.")
        sys.exit(1)

    # Run benchmark
    result = run_benchmark(
        voice_path=args.voice,
        text=args.text,
        config=args.config,
        model_dir=Path(args.model_dir),
        num_runs=args.runs,
        warmup_runs=args.warmup,
        output_path=args.output,
        fp16=args.fp16,
        device=args.device,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        verbose=args.verbose,
    )

    if args.json:
        import json

        output = {
            "startup_time": result.startup_time,
            "warmup_times": result.warmup_times,
            "warmup_durations": result.warmup_durations,
            "inference_times": result.inference_times,
            "inference_durations": result.inference_durations,
            "statistics": {
                "mean": result.mean_inference,
                "std_dev": result.std_inference,
                "min": result.min_inference,
                "max": result.max_inference,
                "median": result.median_inference,
            },
            "rtf": {
                "mean": result.mean_rtf,
                "std_dev": result.std_rtf,
                "min": result.min_rtf,
                "max": result.max_rtf,
                "median": result.median_rtf,
            },
            "config": {
                "text": result.text,
                "text_length": len(result.text),
                "num_runs": result.num_runs,
                "device": args.device,
                "fp16": args.fp16,
                "use_accel": args.use_accel,
                "use_torch_compile": args.use_torch_compile,
                "use_cuda_kernel": args.use_cuda_kernel,
                "use_deepspeed": args.use_deepspeed,
            },
        }
        print(json.dumps(output, indent=2))
    else:
        result.print_report()

    if args.csv:
        import csv
        from datetime import datetime

        file_exists = Path(args.csv).is_file()

        with Path(args.csv).open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "Text",
                    "Text Length",
                    "Voice File",
                    "Device",
                    "FP16",
                    "Use Accel",
                    "Use Torch Compile",
                    "Use CUDA Kernel",
                    "Use DeepSpeed",
                    "Startup Time",
                    "Warmup Runs",
                    "Num Runs",
                    "Mean Inference",
                    "Std Dev",
                    "Min",
                    "Max",
                    "Median",
                    "Mean Output Duration",
                    "Median Output Duration",
                    "Mean RTF",
                    "Std RTF",
                    "Min RTF",
                    "Max RTF",
                    "Median RTF",
                    "Run Times",
                    "Run Durations",
                    "Run RTFs",
                ])

                writer.writerow([
                    datetime.now().isoformat(),
                    result.text,
                    len(result.text),
                    args.voice,
                    args.device or "auto",
                    args.fp16,
                    args.use_accel,
                    args.use_torch_compile,
                    args.use_cuda_kernel,
                    args.use_deepspeed,
                    f"{result.startup_time:.4f}",
                    len(result.warmup_times),
                    result.num_runs,
                    f"{result.mean_inference:.4f}",
                    f"{result.std_inference:.4f}",
                    f"{result.min_inference:.4f}",
                    f"{result.max_inference:.4f}",
                    f"{result.median_inference:.4f}",
                    f"{result.mean_duration:.4f}",
                    f"{result.median_duration:.4f}",
                    f"{result.mean_rtf:.4f}",
                    f"{result.std_rtf:.4f}",
                    f"{result.min_rtf:.4f}",
                    f"{result.max_rtf:.4f}",
                    f"{result.median_rtf:.4f}",
                    str([f"{t:.4f}" for t in result.inference_times]),
                    str([f"{d:.4f}" for d in result.inference_durations]),
                    str([f"{r:.4f}" for r in result.rtf_list]),
                ])
        print(f"Results appended to {args.csv}")


if __name__ == "__main__":
    main()
