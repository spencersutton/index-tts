# pyright: reportUnknownVariableType=false, reportAny=false, reportUnknownArgumentType=false
from pathlib import Path

import safetensors.torch
import torch

base = Path("checkpoints")


def convert_checkpoint(filename: str) -> None:
    new_file = (base / filename).with_suffix(".safetensors")
    if new_file.exists():
        print(f"{new_file} already exists, skipping conversion.")
        return

    model = torch.load(base / filename, map_location="cpu")
    safetensors.torch.save_file(model, new_file)


convert_checkpoint("wav2vec2bert_stats.pt")
convert_checkpoint("gpt.pth")

for filename in ["cfm", "length_regulator", "gpt_layer"]:
    new_file = (base / filename).with_suffix(".safetensors")
    if new_file.exists():
        print(f"{new_file} already exists, skipping conversion.")
        continue

    s2mel = torch.load((base / "s2mel.pth"), map_location="cpu")
    model = s2mel["net"][filename]
    safetensors.torch.save_file(model, new_file)
