# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

from indextts.utils.maskgct.models.codec.amphion_codec.quantize import ResidualVQ
from indextts.utils.maskgct.models.codec.kmeans.vocos import VocosBackbone


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        assert m.bias is not None
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class RepCodec(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(VocosBackbone(), nn.Linear(384, 1024))
        self.quantizer = ResidualVQ()

        self.apply(init_weights)

    def quantize(self, x):
        x = self.encoder(x.mT).mT

        quantized_out = self.quantizer(x)

        return quantized_out.mT
