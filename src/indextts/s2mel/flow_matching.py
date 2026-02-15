import math
from typing import Final, cast, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm

from indextts.s2mel.gpt_fast.model import Transformer
from indextts.util import patch_call

BLOCK_SIZE: Final = 16384
CFG_RATE: Final = 0.7
CHANNELS = 80
DIFFUSION_STEPS: Final = 25
DIM = 512
KERNEL_SIZE: Final = 5
N_LAYERS: Final = 8
STYLE_DIM: Final = 192


class _SConv1d(nn.Module):
    """
    Conv1d layer with built-in handling of asymmetric padding and normalization.
    """

    conv: nn.Conv1d
    kernel_size: int

    @staticmethod
    def _remap_weights(_module: object, state_dict: dict[str, object], *_args: object) -> None:
        for k in list(state_dict.keys()):
            new_k = k.replace("conv.conv", "conv")
            state_dict[new_k] = state_dict.pop(k)

    def __init__(self, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.register_load_state_dict_pre_hook(self._remap_weights)

        self.kernel_size = kernel_size
        self.conv = weight_norm(nn.Conv1d(DIM, out_channels, kernel_size))

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.kernel_size > 1:
            x = F.pad(x, (2, 2), "reflect")
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _WaveNet(nn.Module):
    cond_layer: _SConv1d
    in_layers: nn.ModuleList[_SConv1d]
    res_skip_layers: nn.ModuleList[_SConv1d]

    def __init__(self) -> None:
        super().__init__()

        self.cond_layer = _SConv1d(DIM * 2 * N_LAYERS, 1)
        layers = [_SConv1d(DIM * 2, KERNEL_SIZE) for _ in range(N_LAYERS)]
        self.in_layers = nn.ModuleList(layers)

        layers = [_SConv1d(DIM * 2, 1) for _ in range(N_LAYERS - 1)]
        layers.append(_SConv1d(DIM, 1))
        self.res_skip_layers = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        for i in range(N_LAYERS):
            offset = i * DIM * 2
            g_l = g[:, offset : offset + DIM * 2, :]

            x_in = self.in_layers[i].__call__(x)
            t_act_part, s_act_part = (x_in + g_l).split(DIM, dim=1)
            acts = t_act_part.tanh() * s_act_part.sigmoid()

            res_skip_acts = self.res_skip_layers[i].__call__(acts)
            if i < N_LAYERS - 1:
                res_acts = res_skip_acts[:, :DIM, :]
                x = x + res_acts
                output += res_skip_acts[:, DIM:, :]
            else:
                output += res_skip_acts
        return output

    @patch_call(forward)
    def __call__(self) -> None: ...


class _TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    freqs: Tensor
    mlp: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(DIM // 2, DIM), nn.SiLU(), nn.Linear(DIM, DIM))

        half = DIM // 4
        self.freqs = nn.Buffer((-math.log(10000) * torch.arange(half).float() / half).exp())

    @override
    def forward(self, t: Tensor) -> Tensor:
        args = 1000 * t[:, None] * self.freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    adaLN_modulation: nn.Sequential
    linear: nn.Linear
    norm_final: nn.LayerNorm

    def __init__(self) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(DIM, DIM))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 2 * DIM))

    @override
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _DiT(nn.Module):
    cond_projection: nn.Linear
    cond_x_merge_linear: nn.Linear
    conv1: nn.Linear
    conv2: nn.Conv1d
    final_layer: _FinalLayer
    input_pos: Tensor
    res_projection: nn.Linear
    skip_linear: nn.Linear
    t_embedder: _TimestepEmbedder
    t_embedder2: _TimestepEmbedder
    transformer: Transformer
    wavenet: _WaveNet

    def __init__(self) -> None:
        super().__init__()

        self.cond_projection = nn.Linear(DIM, DIM)  # continuous content
        self.cond_x_merge_linear = nn.Linear(DIM + CHANNELS * 2 + STYLE_DIM, DIM)
        self.conv1 = nn.Linear(DIM, DIM)
        self.conv2 = nn.Conv1d(DIM, CHANNELS, kernel_size=1)
        self.final_layer = _FinalLayer()
        self.input_pos = nn.Buffer(torch.arange(BLOCK_SIZE))
        self.res_projection = nn.Linear(DIM, DIM)
        self.skip_linear = nn.Linear(DIM + CHANNELS, DIM)
        self.t_embedder = _TimestepEmbedder()
        self.t_embedder2 = _TimestepEmbedder()
        self.transformer = Transformer()
        self.wavenet = _WaveNet()

    @override
    def forward(self, x: Tensor, prompt_x: Tensor, t: Tensor, style: Tensor, cond: Tensor) -> Tensor:
        T = x.size(2)

        t1 = self.t_embedder.__call__(t)
        cond = self.cond_projection(cond)

        x = x.mT
        prompt_x = prompt_x.mT

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)
        x_in = self.cond_x_merge_linear(x_in)

        x_res = self.input_pos[: x_in.size(1)]
        x_res = self.transformer.__call__(x_in, t1.unsqueeze(1), x_res)
        x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))

        t2 = self.t_embedder2.__call__(t).unsqueeze(2)
        x = self.conv1(x_res).mT
        x = self.wavenet.__call__(x, g=t2).mT + self.res_projection(x_res)
        x = self.final_layer.__call__(x, t1).mT
        return self.conv2(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class CFM(nn.Module):
    criterion: nn.L1Loss
    estimator: _DiT

    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()
        self.estimator = _DiT()

    @torch.inference_mode()
    def inference(self, mu: Tensor, prompt: Tensor, style: Tensor) -> Tensor:
        """Forward diffusion

        Args:
            mu (Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), DIM)
            prompt (Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (Tensor): reference global style
                shape: (batch_size, STYLE_DIM)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T, _ = mu.shape
        x = torch.randn([B, CHANNELS, T], device=mu.device)
        t_span: Final = torch.linspace(0, 1, DIFFUSION_STEPS + 1, device=mu.device)

        prompt_len: Final = prompt.size(-1)

        # Stack original and CFG (null) inputs for batched processing
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)])
        style = torch.cat([style, torch.zeros_like(style)])
        mu = torch.cat([mu, torch.zeros_like(mu)])

        x[..., :prompt_len] = 0
        t = t_span[0]
        for step in tqdm(range(1, len(t_span))):
            # Perform a single forward pass for both original and CFG inputs
            stacked_dphi_dt = self.estimator.__call__(torch.cat([x, x]), prompt_x, torch.stack([t, t]), style, mu)

            # Split the output back into the original and CFG components
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2)

            # Apply CFG formula
            dphi_dt = (1.0 + CFG_RATE) * dphi_dt - CFG_RATE * cfg_dphi_dt

            dt = t_span[step] - t_span[step - 1]
            x += dt * dphi_dt
            t += dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return x

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization for the estimator model.

        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True  # pyright: ignore[reportPrivateUsage]
        self.estimator = cast(_DiT, torch.compile(self.estimator, fullgraph=True, dynamic=True))
