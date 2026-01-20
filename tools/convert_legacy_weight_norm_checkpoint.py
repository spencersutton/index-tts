"""Convert legacy weight_norm checkpoints to parametrizations-based format.

PyTorch deprecated `nn.utils.weight_norm` in favor of
`nn.utils.parametrizations.weight_norm`.

Legacy checkpoints store weight-norm parameters as:
  - *.weight_g
  - *.weight_v

Parametrizations-based weight norm stores them as:
  - *.parametrizations.weight.original0  (g)
  - *.parametrizations.weight.original1  (v)

This script rewrites checkpoint keys so models that use
`nn.utils.parametrizations.weight_norm` can load without any
runtime key-migration shim.

It supports the IndexTTS s2mel checkpoint format saved by torch.save with:
  {"net": {<submodule>: <state_dict>, ...}, ...}

It can also convert a plain state_dict-like dict.

Usage:
  python tools/convert_legacy_weight_norm_checkpoint.py --in checkpoints/your.pt --out checkpoints/your.converted.pt

Optional:
  --inplace   Overwrite the input file.
  --keep-legacy  Keep legacy *.weight_g/*.weight_v keys in addition to new keys (default: drop legacy keys).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _convert_state_dict_keys(sd: dict[str, Any], *, keep_legacy: bool) -> tuple[dict[str, Any], int]:
    """Convert keys in a state_dict-like mapping.

    Returns: (converted_dict, num_converted_keys)
    """
    converted: dict[str, Any] = dict(sd)
    changed = 0

    for k, v in list(sd.items()):
        if not isinstance(k, str):
            continue
        if k.endswith(".weight_g"):
            prefix = k[: -len(".weight_g")]
            nk = prefix + ".parametrizations.weight.original0"
            if nk not in converted:
                converted[nk] = v
            if not keep_legacy:
                converted.pop(k, None)
            changed += 1
        elif k.endswith(".weight_v"):
            prefix = k[: -len(".weight_v")]
            nk = prefix + ".parametrizations.weight.original1"
            if nk not in converted:
                converted[nk] = v
            if not keep_legacy:
                converted.pop(k, None)
            changed += 1

    return converted, changed


def _convert_checkpoint(obj: Any, *, keep_legacy: bool) -> tuple[Any, int]:
    """Convert a checkpoint object (dict with optional nested state_dicts)."""
    total_changed = 0

    if isinstance(obj, dict) and "net" in obj and isinstance(obj["net"], dict):
        new_obj = dict(obj)
        new_net = dict(obj["net"])
        for name, sub in obj["net"].items():
            if isinstance(sub, dict):
                sub_converted, n = _convert_state_dict_keys(sub, keep_legacy=keep_legacy)
                new_net[name] = sub_converted
                total_changed += n
        new_obj["net"] = new_net
        return new_obj, total_changed

    # Plain state_dict-like dict
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        converted, n = _convert_state_dict_keys(obj, keep_legacy=keep_legacy)
        return converted, n

    raise TypeError("Unsupported checkpoint format. Expected a dict with a 'net' dict or a plain state_dict dict.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input checkpoint path (torch.save format)")
    p.add_argument("--out", dest="out", default=None, help="Output checkpoint path")
    p.add_argument("--inplace", action="store_true", help="Overwrite input file")
    p.add_argument(
        "--keep-legacy",
        action="store_true",
        help="Keep legacy *.weight_g/*.weight_v keys in addition to parametrizations keys",
    )
    args = p.parse_args()

    inp = args.inp
    if args.inplace and args.out is not None:
        raise SystemExit("Use either --inplace or --out, not both")

    if not Path(inp).exists():
        raise SystemExit(f"Input file not found: {inp}")

    out = inp if args.inplace else args.out
    if out is None:
        inp_path = Path(inp)
        root = inp_path.parent / inp_path.stem
        ext = inp_path.suffix
        out = str(root) + ".parametrizations" + ext

    ckpt = torch.load(inp, map_location="cpu")
    converted, n_changed = _convert_checkpoint(ckpt, keep_legacy=args.keep_legacy)

    torch.save(converted, out)

    legacy_note = "(kept legacy keys)" if args.keep_legacy else "(dropped legacy keys)"
    print(f"Converted {n_changed} legacy weight_norm key(s) {legacy_note} -> {out}")


if __name__ == "__main__":
    main()
