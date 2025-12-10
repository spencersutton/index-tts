from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from indextts.config import S2MelConfig


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: Tensor,
    input_b: Tensor,
) -> Tensor:
    in_act = input_a + input_b
    t_act_part, s_act_part = torch.chunk(in_act, 2, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    return t_act * s_act


def sequence_mask(length: Tensor, max_length: Tensor | None = None) -> Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator

    gpt_layer: nn.Sequential | None
    cfm: CFM
    length_regulator: InterpolateRegulator
    models: nn.ModuleDict

    def __init__(self, args: S2MelConfig, use_gpt_latent: bool = False) -> None:
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

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

    def forward(
        self,
        x: Tensor,
        target_lengths: Tensor,
        prompt_len: int,
        cond: Tensor,
        y: Tensor,
    ) -> Tensor:

        return self.cfm(x, target_lengths, prompt_len, cond, y)

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
