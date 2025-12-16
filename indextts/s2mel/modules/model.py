from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from indextts.config import S2MelConfig
from indextts.s2mel.modules.flow_matching import CFM


class MyModel(nn.Module):
    from indextts.s2mel.modules.flow_matching import CFM  # noqa: PLC0415
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator  # noqa: PLC0415

    gpt_layer: nn.Sequential[nn.Module]
    cfm: CFM
    length_regulator: InterpolateRegulator
    models: nn.ModuleDict[CFM | InterpolateRegulator | nn.Sequential[nn.Module]]

    def __init__(self, args: S2MelConfig) -> None:
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM  # noqa: PLC0415
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator  # noqa: PLC0415

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            in_channels=args.length_regulator.in_channels,
            codebook_size=args.length_regulator.content_codebook_size,
        )

        self.cfm = CFM(args)
        self.length_regulator = length_regulator
        self.models = nn.ModuleDict({
            "cfm": self.cfm,
            "length_regulator": self.length_regulator,
        })

        self.gpt_layer = torch.nn.Sequential(
            torch.nn.Linear(1280, 256),
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, 1024),
        )
        self.models["gpt_layer"] = self.gpt_layer

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization.

        This method applies torch.compile to the model for significant
        performance improvements during inference.
        """
        self.cfm.enable_torch_compile()


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    state = cast(
        dict[str, dict[str, dict[str, Tensor]]],
        torch.load(path, map_location="cpu"),
    )

    print(f"{model.__class__.__name__} loaded")
    model.load_state_dict(state, strict=False)
    model.eval()

    return model
