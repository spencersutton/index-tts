import torch
from torch import nn
from tqdm import tqdm

from indextts.s2mel.modules.diffusion_transformer import DiT
from indextts.util import patch_call

SIGMA_MIN = 1e-6
IN_CHANNELS = 80
INFERENCE_CFG_RATE = 0.7


class CFM(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = torch.nn.L1Loss()
        self.estimator = DiT()

    @torch.inference_mode()
    def inference(self, mu: torch.Tensor, prompt: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Forward diffusion

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T, _ = mu.shape
        z = torch.randn([B, IN_CHANNELS, T], device=mu.device)
        t_span = torch.linspace(0, 1, 26, device=mu.device)
        return self.solve_euler(z, prompt, mu, style, t_span)

    def solve_euler(
        self, x: torch.Tensor, prompt: torch.Tensor, mu: torch.Tensor, style: torch.Tensor, t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
        """
        x_lens = torch.tensor([mu.size(1)]).long().to(mu.device)
        t = t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if INFERENCE_CFG_RATE > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)])
                stacked_style = torch.cat([style, torch.zeros_like(style)])
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)])
                stacked_x = torch.cat([x, x])
                stacked_t = torch.stack([t, t])

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt: torch.Tensor = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2)

                # Apply CFG formula
                dphi_dt = (1.0 + INFERENCE_CFG_RATE) * dphi_dt - INFERENCE_CFG_RATE * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]

    def forward(
        self, x1: torch.Tensor, x_lens: torch.Tensor, prompt_lens: torch.Tensor, mu: torch.Tensor, style: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes diffusion loss

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x1: mel
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - SIGMA_MIN) * t) * z + t * x1
        u = x1 - (1 - SIGMA_MIN) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, : prompt_lens[bib]] = x1[bib, :, : prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, : prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib] : x_lens[bib]], u[bib, :, prompt_lens[bib] : x_lens[bib]]
            )
        loss /= b

        return loss, estimator_out + (1 - SIGMA_MIN) * z

    def enable_torch_compile(self):
        """Enable torch.compile optimization for the estimator model.

        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(self.estimator, fullgraph=True, dynamic=True)

    @patch_call(forward)
    def __call__(self): ...
