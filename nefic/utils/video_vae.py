"""Video-VAE wrapper: encode frames to latent / decode latents to frames."""

import torch
import torch.nn as nn
from einops import rearrange
from diffusers import AutoencoderKLCogVideoX


class VideoVAE(nn.Module):
    """
    Wrapper for CogVideoX's 3D causal Video-VAE (E_Vid / D_Vid).

    Provides clean encode/decode interface for single-frame processing,
    handling the temporal dimension internally.
    """

    def __init__(self, vae: AutoencoderKLCogVideoX):
        super().__init__()
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor
        self.vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    @classmethod
    def from_pretrained(cls, model_path, dtype=torch.bfloat16):
        """Load VAE from pretrained CogVideoX model."""
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, subfolder="vae"
        ).to(dtype=dtype)
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()
        return cls(vae)

    @torch.no_grad()
    def encode(self, frames, deterministic=True):
        """
        Encode frames to VAE latent space.

        Args:
            frames: [B, 3, H, W] pixel values in [0, 1]
            deterministic: if True, use .mode(); otherwise .sample()

        Returns:
            latent: [B, 1, C, H/8, W/8] scaled latent
        """
        # Convert to VAE input range [-1, 1] in original precision (float32),
        # then cast to VAE dtype (bfloat16). This matches training/inference reference.
        x = frames * 2.0 - 1.0
        x = x.to(self.vae.dtype)
        x = rearrange(x, 'b c h w -> b c 1 h w')

        latent_dist = self.vae.encode(x).latent_dist
        latent = latent_dist.mode() if deterministic else latent_dist.sample()
        latent = latent * self.scaling_factor

        # [B, C, 1, H/8, W/8] → [B, 1, C, H/8, W/8]
        latent = rearrange(latent, 'b c 1 h w -> b 1 c h w')
        return latent

    @torch.no_grad()
    def decode(self, latent):
        """
        Decode latent to pixel space.

        Args:
            latent: [B, 1, C, H/8, W/8] or [B, T, C, H/8, W/8]

        Returns:
            frames: [B, 3, H, W] pixel values in [0, 1]
        """
        # [B, T, C, H, W] → [B, C, T, H, W]
        x = latent.permute(0, 2, 1, 3, 4)
        x = (1 / self.scaling_factor) * x
        x = x.to(self.vae.dtype)

        frames = self.vae.decode(x).sample
        # [B, C, T, H, W] → take first/only frame
        frames = frames.squeeze(2)
        frames = (frames * 0.5 + 0.5).clamp(0., 1.)
        return frames
