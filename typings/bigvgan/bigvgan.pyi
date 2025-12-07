# pyright: reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false

import torch
from huggingface_hub import PyTorchModelHubMixin

from .env import AttrDict

def load_hparams_from_json(path) -> AttrDict: ...

class AMPBlock1(torch.nn.Module):
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = ...,
        dilation: tuple = ...,
        activation: str = ...,
    ) -> None: ...
    def forward(self, x): ...
    def remove_weight_norm(self) -> None: ...

class AMPBlock2(torch.nn.Module):
    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = ...,
        dilation: tuple = ...,
        activation: str = ...,
    ) -> None: ...
    def forward(self, x): ...
    def remove_weight_norm(self) -> None: ...

class BigVGAN(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url=...,
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = ...) -> None: ...
    def forward(self, x) -> torch.Tensor: ...
    def remove_weight_norm(self) -> None: ...
