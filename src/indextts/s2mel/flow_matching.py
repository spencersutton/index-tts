from typing import Final, cast, override

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm

from indextts.s2mel.audio import N_MELS
from indextts.s2mel.diffusion_transformer import DiT
from indextts.util import patch_call

# Default number of ODE solver steps for the reverse-diffusion process.
# More steps improve quality at the cost of additional forward passes.
DEFAULT_DIFFUSION_STEPS: Final = 25

# Default classifier-free guidance (CFG) interpolation rate.
# Controls how strongly the model conditions on the input vs. an unconditional null baseline.
# Higher values increase adherence to the conditioning signal.
DEFAULT_CFG_RATE: Final = 0.7


class CFM(nn.Module):
    estimator: DiT

    def __init__(self, dim: int = 512) -> None:
        super().__init__()

        self.estimator = DiT(dim)

    @torch.inference_mode()
    @override
    @beartype
    def forward(
        self,
        mu: Float[Tensor, "batch total_time cond_dim"],
        prompt: Float[Tensor, "batch mel_bins prompt_time"],
        style: Float[Tensor, "batch style_dim"],
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        cfg_rate: float = DEFAULT_CFG_RATE,
    ) -> Float[Tensor, "batch mel_bins time"]:
        """Run reverse diffusion (flow matching ODE) to generate a mel-spectrogram.

        Args:
            mu (Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795 + 1069), 512)
            prompt (Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (Tensor): reference global style
                shape: (batch_size, 192)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T, _ = mu.shape
        assert prompt.size(1) == N_MELS, f"Expected prompt to have {N_MELS} mel bins, got {prompt.size(1)}"
        x = torch.randn([B, prompt.size(1), T], device=mu.device)
        t_span: Final = torch.linspace(0, 1, diffusion_steps + 1, device=mu.device)

        prompt_len: Final = prompt.size(-1)

        # Stack original and CFG (null) inputs for batched processing
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt
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
            dphi_dt = (1.0 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt

            dt = t_span[step] - t_span[step - 1]
            x += dt * dphi_dt
            t += dt
            x[:, :, :prompt_len] = 0

        return x

    @patch_call(forward)
    def __call__(self) -> None: ...

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization for the estimator model.

        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True  # pyright: ignore[reportPrivateUsage]
        self.estimator = cast(DiT, torch.compile(self.estimator, fullgraph=True, dynamic=True))
