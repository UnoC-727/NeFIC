"""
NeFIC: Next-Frame Decoding for Ultra-Low-Bitrate Image Compression
with Video Diffusion Priors.

Paper: https://arxiv.org/abs/2603.15129
Code:  https://github.com/UnoC-727/NeFIC
"""

from .codec import NeFICCodec
from .anchor_codec import AnchorCodec
from .generative_decoder import GenerativeDecoder

__all__ = ["NeFICCodec", "AnchorCodec", "GenerativeDecoder"]
