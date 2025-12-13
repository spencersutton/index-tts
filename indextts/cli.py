import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import rich.traceback
import torch

from indextts.infer_v2 import IndexTTS2
from indextts.profiler import dict_calls, profile_func, random_words

if __debug__:
    import omegaconf
    import transformers

    rich.traceback.install(suppress=[omegaconf, torch, transformers], width=120)

# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
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
        default=False,
        help="Enable profiling of function execution times",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=2,
        help="Number of warmup steps before profiling starts",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=5,
        help="Number of active profiling steps",
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
    assert isinstance(args.profile, bool)
    assert isinstance(args.profile_warmup_steps, int)
    assert isinstance(args.profile_steps, int)

    if len(args.text.strip()) == 0 and not args.profile:
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

    voice_file = Path(args.voice)
    text = " ".join(random_words) if args.profile else args.text.strip()

    if args.profile:
        for _ in range(args.profile_warmup_steps):
            random.shuffle(random_words)
            tts.infer(spk_audio_prompt=voice_file, text=text, output_path=output_path)

        sys.setprofile(profile_func)
        for _ in range(args.profile_steps):
            random.shuffle(random_words)
            tts.infer(
                spk_audio_prompt=voice_file,
                text=text,
                output_path=output_path,
                max_text_tokens_per_segment=1500,
            )
        sys.setprofile(None)
    else:
        tts.infer(spk_audio_prompt=voice_file, text=text, output_path=output_path)


if __name__ == "__main__":
    main()
    p = ["location\tfunction_name\tcumulative_time\tcall_count"]
    for code_obj, exec_info in (
        (o, info)
        for (o, info) in sorted(dict_calls.items(), key=lambda item: item[1].cumulative_time, reverse=True)
        if "site-packages" not in o.co_filename and info.cumulative_time >= 0.1
    ):
        p.append(
            f"{code_obj.co_filename}:{code_obj.co_firstlineno}\t"
            f"{code_obj.co_name}\t"
            f"{exec_info.cumulative_time:.6f}s\t"
            f"{exec_info.call_count}"
        )

    for line in p:
        print(line)
    profile_report = "\n".join(p)
    Path(f"function_profile_{time.perf_counter()}.txt").write_text(profile_report)
