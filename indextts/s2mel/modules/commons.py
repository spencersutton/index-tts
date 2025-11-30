import math

import torch
from munch import Munch
from torch import nn
from torch.nn import functional as F


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class AttrDict(UserDict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments_audio(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, _d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    _b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    _b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    # use torch.split to avoid dynamic slicing
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    def __init__(self, args, use_emovec=False, use_gpt_latent=False) -> None:
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels if hasattr(args.length_regulator, "in_channels") else None,
            vector_quantize=args.length_regulator.vector_quantize
            if hasattr(args.length_regulator, "vector_quantize")
            else False,
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=args.length_regulator.n_codebooks if hasattr(args.length_regulator, "n_codebooks") else 1,
            quantizer_dropout=args.length_regulator.quantizer_dropout
            if hasattr(args.length_regulator, "quantizer_dropout")
            else 0.0,
            f0_condition=args.length_regulator.f0_condition
            if hasattr(args.length_regulator, "f0_condition")
            else False,
            n_f0_bins=args.length_regulator.n_f0_bins if hasattr(args.length_regulator, "n_f0_bins") else 512,
        )

        if use_gpt_latent:
            self.models = nn.ModuleDict(
                {
                    "cfm": CFM(args),
                    "length_regulator": length_regulator,
                    "gpt_layer": torch.nn.Sequential(
                        torch.nn.Linear(1280, 256), torch.nn.Linear(256, 128), torch.nn.Linear(128, 1024)
                    ),
                }
            )

        else:
            self.models = nn.ModuleDict({"cfm": CFM(args), "length_regulator": length_regulator})

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

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization.

        This method applies torch.compile to the model for significant
        performance improvements during inference.
        """
        if "cfm" in self.models:
            self.models["cfm"].enable_torch_compile()


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


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
