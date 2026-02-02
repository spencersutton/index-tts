from typing import Final, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm

from indextts.s2mel.modules.diffusion_transformer import DiT


class CFM(nn.Module):
    cfg_rate: Final = 0.7
    criterion: nn.L1Loss
    estimator: DiT

    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()
        self.estimator = DiT()

    @torch.inference_mode()
    def inference(
        self, mu: Float[Tensor, "b t c"], prompt: Float[Tensor, "b c t"], style: Float[Tensor, "b c"]
    ) -> Tensor:
        """Forward diffusion

        Args:
            mu (Tensor): semantic info of reference audio and altered audio
            prompt (Tensor): reference mel
            style (Tensor): reference global style
        Returns:
            sample: generated mel-spectrogram
        """
        B, T, _ = mu.shape
        z = torch.randn([B, 80, T], device=mu.device)
        t_span = torch.linspace(0, 1, 26, device=mu.device)
        return self.solve_euler(z, prompt, mu, style, t_span)

    def solve_euler(
        self,
        x: Float[Tensor, "b c t"],
        prompt: Float[Tensor, "b c t"],
        mu: Float[Tensor, "b t c"],
        style: Float[Tensor, "b c"],
        t_span: Float[Tensor, "t"],  # noqa: UP037
    ) -> Tensor:
        """
        Fixed euler solver for ODEs.
        Args:
            x (Tensor): random noise
            t_span (Tensor): n_timesteps interpolated
            mu (Tensor): semantic info of reference audio and altered audio
            x_lens (Tensor): mel frames output
            prompt (Tensor): reference mel
            style (Tensor): reference global style
        """
        t = t_span[0]

        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0

        # Stack original and CFG (null) inputs for batched processing
        stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)])
        stacked_style = torch.cat([style, torch.zeros_like(style)])
        stacked_mu = torch.cat([mu, torch.zeros_like(mu)])

        for step in tqdm(range(1, len(t_span))):
            # Perform a single forward pass for both original and CFG inputs
            stacked_dphi_dt = self.estimator.__call__(
                torch.cat([x, x]), stacked_prompt_x, torch.stack([t, t]), stacked_style, stacked_mu
            )

            # Split the output back into the original and CFG components
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2)

            # Apply CFG formula
            dphi_dt = 1.7 * dphi_dt - 0.7 * cfg_dphi_dt

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
