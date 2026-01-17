from typing import Any

import torch
from torch import Tensor, nn


@torch.compile
def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: Tensor) -> Tensor:
    n_channels_int = int(n_channels[0])
    in_act = input_a + input_b
    # use torch.split to avoid dynamic slicing
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    return t_act * s_act


def sequence_mask(length: Tensor | int, max_length: int | None = None) -> Tensor:
    length = torch.as_tensor(length)
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    from indextts.s2mel.modules.flow_matching import CFM

    cfm: CFM

    def __init__(self) -> None:
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

        self.cfm = CFM()
        self.length_regulator = InterpolateRegulator()
        self.gpt_layer = nn.Sequential(nn.Linear(1280, 256), nn.Linear(256, 128), nn.Linear(128, 1024))

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization.

        This method applies torch.compile to the model for significant
        performance improvements during inference.
        """
        self.cfm.enable_torch_compile()


def load_checkpoint2(
    model, optimizer, path, load_only_params=True, ignore_modules=[], is_distributed=False, load_ema=False
) -> MyModel:
    state: dict[str, Any] = torch.load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        print("Loading EMA")
        for key in model.models:
            i = 0
            for param_name in params[key]:
                if "input_pos" in param_name:
                    continue
                assert params[key][param_name].shape == state["ema"][key][0][i].shape
                params[key][param_name] = state["ema"][key][0][i].clone()
                i += 1
    for key in model.models:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                # strip prefix of DDP (module.), create a new OrderedDict that does not contain the prefix
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            model_state_dict = model.models[key].state_dict()
            # 过滤出形状匹配的键值对
            filtered_state_dict = {
                k: v for k, v in params[key].items() if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                print(f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}")
            print(f"{key} loaded")
            model.models[key].load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])

    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters
