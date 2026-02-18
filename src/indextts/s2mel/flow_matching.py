from typing import Final, cast

import torch
from torch import Tensor, nn
from tqdm import tqdm

from indextts.s2mel.diffusion_transformer import DiT

CFG_RATE: Final = 0.7
CHANNELS = 80
DIFFUSION_STEPS: Final = 25
DIM = 512


class CFM(nn.Module):
    criterion: nn.L1Loss
    estimator: DiT

    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()
        self.estimator = DiT()

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
        self.estimator = cast(DiT, torch.compile(self.estimator, fullgraph=True, dynamic=True))
