from __future__ import annotations

from pathlib import Path
from typing import cast, override

import torch
from torch import Tensor, nn

from indextts.config import S2MelConfig
from indextts.util import patch_call


class MyModel(nn.Module):
    from indextts.s2mel.modules.flow_matching import CFM  # noqa: PLC0415
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator  # noqa: PLC0415

    gpt_layer: nn.Sequential[[Tensor], Tensor] | None
    cfm: CFM
    length_regulator: InterpolateRegulator
    models: nn.ModuleDict[CFM | InterpolateRegulator | nn.Sequential[[Tensor], Tensor]]

    def __init__(self, args: S2MelConfig, use_gpt_latent: bool = False) -> None:
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

        if use_gpt_latent:
            self.gpt_layer = torch.nn.Sequential(
                torch.nn.Linear(1280, 256),
                torch.nn.Linear(256, 128),
                torch.nn.Linear(128, 1024),
            )
            self.models["gpt_layer"] = self.gpt_layer

    @override
    def forward(
        self,
        x: Tensor,
        target_lengths: Tensor,
        prompt_len: Tensor,
        cond: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.cfm(x, target_lengths, prompt_len, cond, y)

    @patch_call(forward)
    def __call__(self) -> None: ...

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization.

        This method applies torch.compile to the model for significant
        performance improvements during inference.
        """
        self.cfm.enable_torch_compile()


def load_checkpoint(model: MyModel, path: Path) -> MyModel:
    state = cast(
        dict[str, dict[str, dict[str, Tensor]]],
        torch.load(path, map_location="cpu"),
    )
    params = state["net"]

    for key, module in model.models.items():
        if key not in params:
            continue

        state_dict = params[key]
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model_state = module.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        skipped = set(state_dict) - set(filtered)
        if skipped:
            print(f"Warning: Skipped loading keys due to shape mismatch: {skipped}")
        print(f"{key} loaded")
        module.load_state_dict(filtered, strict=False)
    model.eval()

    return model
