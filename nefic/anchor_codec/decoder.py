"""
Anchor Decoder (E_Dec) and Bypass Refinement module.

The decoder reconstructs the anchor frame and produces intermediate features
for bypass refinement, which maps to the video diffusion latent space.
"""

import torch
import torch.nn as nn

from compressai.layers import subpel_conv3x3

from .layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualTokenProjectionBlock,
)


class AnchorDecoder(nn.Module):
    """
    Anchor Decoder: synthesis transform that outputs both
    the reconstructed anchor frame and intermediate features.

    Corresponds to E_Dec in the paper.
    The intermediate features (mid_feat) are used by BypassRefinement
    to produce z_bypass for one-step video diffusion initialization.
    """

    def __init__(self, N=192, M=320):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, y_hat):
        """
        Args:
            y_hat: quantized latent [B, M, H/16, W/16]

        Returns:
            anchor_frame: reconstructed anchor [B, 3, H, W]
            mid_feat: intermediate feature [B, N, H/4, W/4] for bypass refinement
        """
        x1 = self.synthesis_transform[0](y_hat)    # ResidualBlock: M → N
        mid_feat = self.synthesis_transform[1](x1)  # ResidualBlockUpsample: 2x up
        anchor_frame = self.synthesis_transform[2:](mid_feat)
        return anchor_frame, mid_feat


class BypassRefinement(nn.Module):
    """
    Bypass Refinement module T: maps anchor codec intermediate features
    to the video diffusion latent space.

    Corresponds to the Semantic Bypass Refinement in Sec. 3.5.
    Produces z_bypass = T(h_dec) which serves as the initial noisy latent
    for one-step generation, replacing pure Gaussian noise.

    """

    def __init__(self, N=192, out_channels=16):
        super().__init__()
        num_projection_layers = 3
        projection_channels = 320

        self.project_in = nn.Linear(N, projection_channels)
        self.projection_layers = nn.Sequential(
            *[ResidualTokenProjectionBlock(projection_channels)
              for _ in range(num_projection_layers)]
        )
        self.project_out = nn.Linear(projection_channels, out_channels)

    def forward(self, mid_feat):
        """
        Args:
            mid_feat: intermediate features from AnchorDecoder [B, N, H, W]

        Returns:
            bypass_latent: [B, 1, out_channels, H, W] — initialization for VDM
        """
        B, C, H, W = mid_feat.shape
        feat = mid_feat.view(B, C, H * W).transpose(1, 2)
        feat = self.project_in(feat)
        feat = self.projection_layers(feat)
        feat = self.project_out(feat).transpose(1, 2).view(B, -1, H, W)
        # Add frame dimension for VDM compatibility
        bypass_latent = feat.unsqueeze(1)
        return bypass_latent
