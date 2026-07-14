"""
Anchor Encoder (E_Enc): Analysis transform with conditional encoding.

Conditioned on Video-VAE latents for latent-space alignment between
the compression domain and the video generation domain.
"""

import torch
import torch.nn as nn

from .layers import (
    ResidualBlockWithStride,
    ResidualBlock,
    ResidualTokenProjectionBlock,
    conv3x3,
)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class AnchorEncoder(nn.Module):
    """
    Anchor Encoder with conditional encoding on Video-VAE latent.

    Corresponds to E_Enc in the paper. The encoder conditions on the
    Video-VAE latent z_0 = E_Vid(x) to align the compression latent
    space with the video generation space (Sec. 3.5, Conditional Anchor Encoding).

    Architecture:
        - Standard analysis transform (downsample 16x)
        - At 8x downscale, fuse with VAE condition via transformer
    """

    def __init__(self, N=192, M=320):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

        # Parameter names match checkpoint layout: project_in_en, information_transformer_layes_en, spatial_ch_projs_en
        num_projection_layers = 3
        num_proj_channel = 16  # VAE condition channel dim
        projection_channels = 320

        self.project_in_en = nn.Linear(N + num_proj_channel, projection_channels)
        self.information_transformer_layes_en = nn.Sequential(
            *[ResidualTokenProjectionBlock(projection_channels)
              for _ in range(num_projection_layers)]
        )
        self.spatial_ch_projs_en = zero_module(nn.Linear(projection_channels, N))

    def _conditional_fusion(self, mid_feat):
        B, C, H, W = mid_feat.shape
        feat = mid_feat.view(B, C, H * W).transpose(1, 2)
        feat = self.project_in_en(feat)
        feat = self.information_transformer_layes_en(feat)
        feat = self.spatial_ch_projs_en(feat).transpose(1, 2).view(B, 192, H, W)
        return feat

    def forward(self, x, vae_condition):
        """
        Args:
            x: input image [B, 3, H, W]
            vae_condition: Video-VAE latent [B, 16, H/8, W/8]

        Returns:
            y: compressed latent [B, M, H/16, W/16]
        """
        # Process through first 5 layers (down to 8x)
        x0 = self.analysis_transform[0:5](x)

        # Conditional fusion: concatenate with VAE latent and apply transformer
        x_mid = torch.cat((x0, vae_condition), dim=1)
        x_mid_add = self._conditional_fusion(x_mid)
        x0 = x0 + x_mid_add

        # Continue through remaining layers (8x → 16x)
        y = self.analysis_transform[5:](x0)
        return y
