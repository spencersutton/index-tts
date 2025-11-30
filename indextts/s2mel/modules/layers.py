import random

import torch
import torch.nn.functional as F
import torchaudio.functional as audio_F
from torch import nn

random.seed(0)


def _get_activation_fn(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == "lrelu":
        return nn.LeakyReLU(0.2)
    elif activ == "swish":
        return lambda x: x * torch.sigmoid(x)
    else:
        raise RuntimeError(f"Unexpected activ type {activ}, expected [relu, lrelu, swish]")


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear") -> None:
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        param=None,
    ) -> None:
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class CausualConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        param=None,
    ) -> None:
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, : -self.padding]
        return x


class CausualBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ="lrelu") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p) for i in range(n_conv)]
        )

    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ="lrelu", dropout_p=0.2):
        layers = [
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_p),
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p),
        ]
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ="relu") -> None:
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList(
            [self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p) for i in range(n_conv)]
        )

    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ="relu", dropout_p=0.2):
        layers = [
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p),
        ]
        return nn.Sequential(*layers)
