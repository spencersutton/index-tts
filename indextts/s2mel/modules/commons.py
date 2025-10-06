import argparse

import numpy as np
import torch
from torch import nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import logging

        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def normalize_f0(f0_sequence):
    # Remove unvoiced frames (replace with -1)
    voiced_indices = np.where(f0_sequence > 0)[0]
    f0_voiced = f0_sequence[voiced_indices]

    # Convert to log scale
    log_f0 = np.log2(f0_voiced)

    # Calculate mean and standard deviation
    mean_f0 = np.mean(log_f0)
    std_f0 = np.std(log_f0)

    # Normalize the F0 sequence
    normalized_f0 = (log_f0 - mean_f0) / std_f0

    # Create the normalized F0 sequence with unvoiced frames
    normalized_sequence = np.zeros_like(f0_sequence)
    normalized_sequence[voiced_indices] = normalized_f0
    normalized_sequence[f0_sequence <= 0] = -1  # Assign -1 to unvoiced frames

    return normalized_sequence


class MyModel(nn.Module):
    def __init__(self, args, use_emovec=False, use_gpt_latent=False):
        super(MyModel, self).__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels
            if hasattr(args.length_regulator, "in_channels")
            else None,
            vector_quantize=args.length_regulator.vector_quantize
            if hasattr(args.length_regulator, "vector_quantize")
            else False,
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=args.length_regulator.n_codebooks
            if hasattr(args.length_regulator, "n_codebooks")
            else 1,
            quantizer_dropout=args.length_regulator.quantizer_dropout
            if hasattr(args.length_regulator, "quantizer_dropout")
            else 0.0,
            f0_condition=args.length_regulator.f0_condition
            if hasattr(args.length_regulator, "f0_condition")
            else False,
            n_f0_bins=args.length_regulator.n_f0_bins
            if hasattr(args.length_regulator, "n_f0_bins")
            else 512,
        )

        if use_gpt_latent:
            self.models = nn.ModuleDict(
                {
                    "cfm": CFM(args),
                    "length_regulator": length_regulator,
                    "gpt_layer": torch.nn.Sequential(
                        torch.nn.Linear(1280, 256),
                        torch.nn.Linear(256, 128),
                        torch.nn.Linear(128, 1024),
                    ),
                }
            )

        else:
            self.models = nn.ModuleDict(
                {"cfm": CFM(args), "length_regulator": length_regulator}
            )

    def forward(self, x, target_lengths, prompt_len, cond, y):
        x = self.models["cfm"](x, target_lengths, prompt_len, cond, y)
        return x

    def forward2(self, S_ori, target_lengths, F0_ori):
        x = self.models["length_regulator"](S_ori, ylens=target_lengths, f0=F0_ori)
        return x

    def forward_emovec(self, x):
        x = self.models["emo_layer"](x)
        return x

    def forward_emo_encoder(self, x):
        x = self.models["emo_encoder"](x)
        return x

    def forward_gpt(self, x):
        x = self.models["gpt_layer"](x)
        return x


def load_checkpoint2(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
    load_ema=False,
):
    state = torch.load(path, map_location="cpu")
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
                k: v
                for k, v in params[key].items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                print(
                    f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
                )
            print("%s loaded" % key)
            model.models[key].load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    #     _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])

    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters
