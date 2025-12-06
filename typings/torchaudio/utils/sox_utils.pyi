
from torchaudio._internal.module_utils import dropping_support

"""Module to change the configuration of libsox, which is used by I/O functions like
:py:mod:`~torchaudio.backend.sox_io_backend` and :py:mod:`~torchaudio.sox_effects`.

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:

    - Some APIs are deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.
"""
sox_ext = ...

@dropping_support
def set_seed(seed: int):  # -> None:
    ...
@dropping_support
def set_verbosity(verbosity: int):  # -> None:
    ...
@dropping_support
def set_buffer_size(buffer_size: int):  # -> None:
    ...
@dropping_support
def set_use_threads(use_threads: bool):  # -> None:
    ...
@dropping_support
def list_effects() -> dict[str, str]: ...
@dropping_support
def list_read_formats() -> list[str]: ...
@dropping_support
def list_write_formats() -> list[str]: ...
@dropping_support
def get_buffer_size() -> int: ...
