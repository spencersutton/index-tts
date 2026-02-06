# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        weight_norm(m)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath: str | Path, device: str | torch.device) -> dict[str, object]:
    filepath = Path(filepath)
    assert filepath.is_file()
    print(f"Loading '{filepath}'")
    checkpoint_dict = cast(dict[str, object], torch.load(filepath, map_location=device, weights_only=True))
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath: str | Path, obj: object) -> None:
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir: str | Path, prefix: str, renamed_file: str | None = None) -> str | None:
    # Fallback to original scanning logic first
    cp_dir_path = Path(cp_dir)
    cp_list = sorted(cp_dir_path.glob(prefix + "????????"))

    if len(cp_list) > 0:
        last_checkpoint_path = max(cp_list)
        print(f"[INFO] Resuming from checkpoint: '{last_checkpoint_path}'")
        return str(last_checkpoint_path)

    # If no pattern-based checkpoints are found, check for renamed file
    if renamed_file:
        renamed_path = cp_dir_path / renamed_file
        if renamed_path.is_file():
            print(f"[INFO] Resuming from renamed checkpoint: '{renamed_file}'")
            return str(renamed_path)

    return None
