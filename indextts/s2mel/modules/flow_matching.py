from abc import ABC

import torch
from torch import Tensor, nn

from indextts.config import S2MelConfig
from indextts.s2mel.modules.diffusion_transformer import DiT


class CFM(nn.Module, ABC):
    estimator: DiT
    criterion = nn.L1Loss()
    in_channels: int
    sigma_min = 1e-6

    def __init__(self, args: S2MelConfig) -> None:
        super().__init__()

        self.estimator = DiT(args)
        self.in_channels = args.DiT.in_channels

    @torch.inference_mode()
    def inference(
        self,
        mu: Tensor,
        x_lens: Tensor,
        prompt: Tensor,
        style: Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
    ) -> Tensor:
        """Forward diffusion.

        Args:
            mu (Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (Tensor): reference global style
                shape: (batch_size, 192)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate)

    def solve_euler(
        self,
        x: Tensor,
        x_lens: Tensor,
        prompt: Tensor,
        mu: Tensor,
        style: Tensor,
        t_span: Tensor,
        inference_cfg_rate: float = 0.5,
    ) -> Tensor:
        """
        Fixed euler solver for ODEs.
        Args:
            x (Tensor): random noise
            t_span (Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (Tensor): reference global style
                shape: (batch_size, 192)
        """
        t = t_span[0]

        # Pre-compute prompt masking
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0

        # Pre-allocate CFG tensors (reused each iteration to avoid allocation overhead)
        null_prompt_x = torch.zeros_like(prompt_x)
        null_style = torch.zeros_like(style)
        null_mu = torch.zeros_like(mu)

        # Pre-compute CFG scale factors
        cfg_scale = 1.0 + inference_cfg_rate
        neg_cfg_rate = -inference_cfg_rate

        n_steps = len(t_span) - 1
        for step in range(n_steps):
            dt = t_span[step + 1] - t_span[step]

            # Stack original and CFG (null) inputs for batched processing
            stacked_prompt_x = torch.cat([prompt_x, null_prompt_x], dim=0)
            stacked_style = torch.cat([style, null_style], dim=0)
            stacked_mu = torch.cat([mu, null_mu], dim=0)
            stacked_x = torch.cat([x, x], dim=0)
            stacked_t = t.unsqueeze(0).expand(2)

            # Perform a single forward pass for both original and CFG inputs
            stacked_dphi_dt = self.estimator(
                stacked_x,
                stacked_prompt_x,
                x_lens,
                stacked_t,
                stacked_style,
                stacked_mu,
            )

            # Split the output back into the original and CFG components
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

            # Apply CFG formula: (1 + cfg_rate) * dphi - cfg_rate * cfg_dphi
            # Fused operation to avoid intermediate tensor allocation
            dphi_dt = cfg_scale * dphi_dt + neg_cfg_rate * cfg_dphi_dt

            # In-place update
            x += dt * dphi_dt
            t += dt
            x[:, :, :prompt_len] = 0

        return x
