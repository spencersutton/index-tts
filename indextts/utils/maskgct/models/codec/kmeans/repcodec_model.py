# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from torch.nn import functional as F

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
    codebook_size = 8192
    codebook_dim = 8
    hidden_size = 1024
    vocos_dim = 384
    vocos_intermediate_dim = 2048
    vocos_num_layers = 12
    num_quantizers = 1
    downsample_scale = 1

    def __init__(self) -> None:
        super().__init__()

        if self.downsample_scale is not None and self.downsample_scale > 1:
            self.down = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1)
            self.up = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1)

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=self.hidden_size,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # downsample
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        # encoder
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        # vq
        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        # decoder
        x = self.decoder(quantized_out)

        # up
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x_rec = self.up(x).transpose(1, 2)

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices

        return x_rec, codebook_loss, all_indices

    def quantize(self, x):

        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

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
