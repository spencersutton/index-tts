import torch

from indextts.s2mel.modules.audio import get_mel_and_window


def test_get_mel_and_window_respects_device_and_dtype() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    mel, window = get_mel_and_window(device, dtype)

    assert mel.device == device
    assert window.device == device
    assert mel.dtype == dtype
    assert window.dtype == dtype
