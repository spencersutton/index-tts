from pathlib import Path

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

base = Path("checkpoints")


def convert_checkpoint(filename: str) -> None:
    new_file = (base / filename).with_suffix(".safetensors")
    if new_file.exists():
        print(f"{new_file} already exists, skipping conversion.")
        return

    model = torch.load(base / filename, map_location="cpu")
    safetensors.torch.save_file(model, new_file)
    print(f"Converted {filename} to safetensors.")


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
    print(f"Converted {filename} to safetensors.")

convert_checkpoint("wav2vec2bert_stats.pt")

if not Path("checkpoints/campplus_cn_common.safetensors").exists():
    checkpoint = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    model = torch.load(checkpoint, map_location="cpu")
    safetensors.torch.save_file(model, "checkpoints/campplus_cn_common.safetensors")
    print("Converted campplus_cn_common.bin to safetensors.")
else:
    print("checkpoints/campplus_cn_common.safetensors already exists, skipping conversion.")
