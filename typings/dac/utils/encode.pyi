import argbind
import torch

@argbind.bind(group="encode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    output: str = ...,
    weights_path: str = ...,
    model_tag: str = ...,
    model_bitrate: str = ...,
    n_quantizers: int = ...,
    device: str = ...,
    model_type: str = ...,
    win_duration: float = ...,
    verbose: bool = ...,
):  # -> None:
    ...

if __name__ == "__main__":
    args = ...
