"""Attention modules for the Anchor Codec."""

import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict

from compressai.layers import conv3x3
from .conv import conv1x1


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def build_position_index(window_size):
    """Build relative position index for window attention."""
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves input dtype."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Fast approximation of the GELU activation function."""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualTokenProjectionBlock(nn.Module):
    """Residual block with token-wise projection and MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 2)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 2, d_model)),
            ])
        )
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        projected = self.output_proj(self.value_proj(self.ln_1(x)))
        x = x + projected
        x = x + self.mlp(self.ln_2(x))
        return x


class AttentionBlock(nn.Module):
    """Self-attention block for image compression (from Cheng et al. 2020)."""

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
