from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from bigvgan.bigvgan import BigVGAN


class BigVGANHFModel(StrEnum):
    """
    BigVGAN HF models.
    """

    V2_44KHZ_128BAND_512X = "nvidia/bigvgan_v2_44khz_128band_512x"
    V2_44KHZ_128BAND_256X = "nvidia/bigvgan_v2_44khz_128band_256x"
    V2_24KHZ_100BAND_256X = "nvidia/bigvgan_v2_24khz_100band_256x"
    V2_22KHZ_80BAND_256X = "nvidia/bigvgan_v2_22khz_80band_256x"
    V2_22KHZ_80BAND_FMAX8K_256X = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
    V2_24KHZ_100BAND = "nvidia/bigvgan_24khz_100band"
    V2_22KHZ_80BAND = "nvidia/bigvgan_22khz_80band"
    BASE_24KHZ_100BAND = "nvidia/bigvgan_base_24khz_100band"
    BASE_22KHZ_80BAND = "nvidia/bigvgan_base_22khz_80band"

    def __str__(self) -> str:
        return self.value


class BigVGANInference(BigVGAN):
    """
    BigVGAN inference.
    """

    def __init__(self, h: dict[str, object], use_cuda_kernel: bool = False) -> None:
        super().__init__(h, use_cuda_kernel)

        # set to eval and remove weight norm
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with inference mode enabled.

        Args:
            mel (torch.Tensor): Input mel spectrogram

        Returns:
            torch.Tensor: Generated audio waveform
        """
        with torch.inference_mode():
            return super().forward(x)

    if TYPE_CHECKING:

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward(x)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        model = super()._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            use_cuda_kernel=use_cuda_kernel,
            **model_kwargs,
        )

        # remove weight norm for inference
        model.remove_weight_norm()

        return model
