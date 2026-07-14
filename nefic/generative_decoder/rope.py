"""3D Rotary Position Embedding (RoPE) for CogVideoX transformer."""

from typing import Tuple, Optional

import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = 1,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare 3D rotary positional embeddings for the DiT transformer.

    Args:
        height, width: spatial dimensions of the input
        num_frames: number of temporal frames
        vae_scale_factor_spatial: spatial downscaling of VAE
        patch_size: spatial patch size for patchification
        patch_size_t: temporal patch size
        attention_head_dim: dimension per attention head
        device: target device

    Returns:
        (freqs_cos, freqs_sin): rotary embedding tensors
    """
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )
    return freqs_cos, freqs_sin
