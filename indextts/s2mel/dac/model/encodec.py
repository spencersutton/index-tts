# Shim: re-exported from PyPI `encodec` (facebookresearch/encodec, MIT license).
from encodec.modules import (  # noqa: F401
    NormConv1d, NormConv2d, NormConvTranspose1d, NormConvTranspose2d,
    SConv1d, SConvTranspose1d, SLSTM,
    pad1d, unpad1d,
)
from encodec.modules.conv import (  # noqa: F401
    ConvLayerNorm,
    apply_parametrization_norm, get_norm_module,
    get_extra_padding_for_conv1d, pad_for_conv1d,
)
