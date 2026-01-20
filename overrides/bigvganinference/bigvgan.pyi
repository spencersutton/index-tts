from indextts.util import patch_call
import torch
from _typeshed import Incomplete
from bigvganinference.env import AttrDict as AttrDict
from bigvganinference.utils import get_padding as get_padding, init_weights as init_weights
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

def load_hparams_from_json(path) -> AttrDict: ...

class AMPBlock1(torch.nn.Module):
    h: Incomplete
    convs1: Incomplete
    convs2: Incomplete
    num_layers: Incomplete
    activations: Incomplete
    def __init__(
        self, h: AttrDict, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5), activation: str = None
    ) -> None: ...
    def forward(self, x): ...
    def remove_weight_norm(self) -> None: ...

class AMPBlock2(torch.nn.Module):
    h: Incomplete
    convs: Incomplete
    num_layers: Incomplete
    activations: Incomplete
    def __init__(
        self, h: AttrDict, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5), activation: str = None
    ) -> None: ...
    def forward(self, x): ...
    def remove_weight_norm(self) -> None: ...

class BigVGAN(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    h: Incomplete
    num_kernels: Incomplete
    num_upsamples: Incomplete
    conv_pre: Incomplete
    ups: Incomplete
    resblocks: Incomplete
    activation_post: Incomplete
    use_bias_at_final: Incomplete
    conv_post: Incomplete
    use_tanh_at_final: Incomplete
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def remove_weight_norm(self) -> None: ...
    @patch_call(forward)
    def __call__(self) -> None: ...
