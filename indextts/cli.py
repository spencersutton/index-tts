import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS Command Line")
    parser.add_argument("text", type=str, help="Text to be synthesized")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to the audio prompt file (wav format)")
    parser.add_argument("-o", "--output_path", type=str, default="gen.wav", help="Path to the output wav file")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the config file. Default is 'checkpoints/config.yaml'",
    )
    parser.add_argument(
        "--model_dir", type=str, default="checkpoints", help="Path to the model directory. Default is 'checkpoints'"
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
    parser.add_argument(
        "-f", "--force", action="store_true", default=False, help="Force to overwrite the output file if it exists"
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to run the model on (cpu, cuda, mps, xpu)."
    )
    parser.add_argument(
        "--use_accel", action="store_true", default=False, help="Enable acceleration engine for GPT2 (experimental)."
    )
    args = parser.parse_args()
    voice_file = Path(args.voice)
    output_path = Path(args.output_path)
    model_dir = Path(args.model_dir)

    if len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        parser.print_help()
        sys.exit(1)
    if not voice_file.exists():
        print(f"Audio prompt file {voice_file} does not exist.")
        parser.print_help()
        sys.exit(1)

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
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # pyright: ignore[reportAttributeAccessIssue]
            args.device = "xpu"
        elif hasattr(torch, "mps") and torch.mps.is_available():  # pyright: ignore[reportAttributeAccessIssue]
            args.device = "mps"
        else:
            args.device = "cpu"
            args.fp16 = False  # Disable FP16 on CPU
            print("WARNING: Running on CPU may be slow.")

    print("Importing IndexTTS2...")
    from indextts.infer_v2 import IndexTTS2

    print("Initializing IndexTTS2...")
    tts = IndexTTS2(model_dir=model_dir, use_fp16=args.fp16, device=args.device, use_accel=args.use_accel)
    print("Start inference...")
    tts.infer(output_path=output_path, spk_audio_prompt=voice_file, text=args.text.strip())


if __name__ == "__main__":
    main()
