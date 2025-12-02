import argbind
import torch

@argbind.bind(group="decode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def decode(
    input: str,
    output: str = ...,
    weights_path: str = ...,
    model_tag: str = ...,
    model_bitrate: str = ...,
    device: str = ...,
    model_type: str = ...,
    verbose: bool = ...,
):  # -> None:
    ...

if __name__ == "__main__":
    args = ...
