# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torch import Tensor, nn

from indextts.utils.maskgct.models.codec.amphion_codec.quantize import (
    ResidualVQ,
)
from indextts.utils.maskgct.models.codec.kmeans.vocos import VocosBackbone


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        assert m.bias is not None
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        assert m.bias is not None
        nn.init.constant_(m.bias, 0)


codebook_size = 8192
codebook_dim = 8
hidden_size = 1024
vocos_dim = 384
vocos_intermediate_dim = 2048
vocos_num_layers = 12
num_quantizers = 1
downsample_scale = 1


class RepCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )

        self.quantizer = ResidualVQ()

        self.reset_parameters()

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder.forward(x.transpose(1, 2)).transpose(1, 2)

        (
            quantized_out,
            all_indices,
            _all_commit_losses,
            _all_codebook_losses,
            _,
        ) = self.quantizer(x)

        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self) -> None:
        self.apply(init_weights)
