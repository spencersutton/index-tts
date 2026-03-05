import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import pyinstrument
import torch

from indextts.profiling import generate_profile_report

logger = logging.getLogger(__name__)


def main() -> None:
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
        "-f", "--force", action="store_true", default=False, help="Force to overwrite the output file if it exists"
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to run the model on (cpu, cuda, mps, xpu)."
    )
    parser.add_argument(
        "--use_accel", action="store_true", default=False, help="Enable acceleration engine for GPT2 (experimental)."
    )
    parser.add_argument("--profile", action="store_true", default=False, help="Enable profiling")

    args = parser.parse_args()
    voice_file = Path(cast(str, args.voice))
    output_path = Path(cast(str, args.output_path))

    args.text = cast(str, args.text)
    args.force = cast(bool, args.force)
    args.device = cast(str | None, args.device)
    args.profile = cast(bool, args.profile)
    args.use_accel = cast(bool, args.use_accel)

    if len(args.text.strip()) == 0:
        logger.error("Text is empty.")
        parser.print_help()
        sys.exit(1)
    if not voice_file.exists():
        logger.error("Audio prompt file %s does not exist.", voice_file)
        parser.print_help()
        sys.exit(1)

    if output_path.exists():
        if not args.force:
            logger.error("Output file %s already exists. Use --force to overwrite.", output_path)
            parser.print_help()
            sys.exit(1)
        else:
            output_path.unlink()

    if args.device is None:
        args.device = torch.accelerator.current_accelerator() or torch.get_default_device()

    logger.info("Importing IndexTTS2...")
    from indextts.infer_v2 import IndexTTS2

    torch.set_default_device(args.device)

    profiler = pyinstrument.Profiler()
    if args.profile:
        profiler.start()

    logger.info("Initializing IndexTTS2...")
    tts = IndexTTS2(device=args.device, use_accel=args.use_accel)
    logger.info("Start inference...")
    tts.infer(output_path=output_path, spk_audio_prompt=voice_file, text=args.text.strip())

    if args.profile:
        profiler.stop()
        generate_profile_report(profiler)


if __name__ == "__main__":
    main()
