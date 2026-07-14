"""
NeFICCodec: Unified codec for ultra-low-bitrate image compression
with next-frame video diffusion decoding.

This is the main entry point for NeFIC inference.
"""

import math
import logging

import torch
import torch.nn as nn
from torchvision import transforms

from .anchor_codec import AnchorCodec
from .generative_decoder import GenerativeDecoder
from .utils.video_vae import VideoVAE
from .utils.text_encoder import PromptEncoder
from .utils.color_fix import adain_color_fix_quant
from .utils.image_utils import pad_to_multiple, crop_to_original

logger = logging.getLogger(__name__)


class NeFICCodec(nn.Module):
    """
    NeFIC: Next-Frame Decoding for Ultra-Low-Bitrate Image Compression.

    A unified codec that:
      - Encodes images via a VAE-conditioned Anchor Codec
      - Decodes via one-step video diffusion generation (CogVideoX + LoRA)

    The bypass latent (z_bypass) is generated internally as a byproduct of
    the anchor codec and is not exposed to external callers.

    Usage:
        codec = NeFICCodec.from_pretrained(model_path, checkpoint_dir)
        result = codec(image_tensor)
        # result["x_hat"] is the reconstructed image
        # result["bpp"] is the estimated bitrate
    """

    def __init__(self, anchor_codec, generative_decoder, video_vae, prompt_encoder,
                 pad_multiple=64, with_color_fix=True):
        super().__init__()
        self.anchor_codec = anchor_codec
        self.generative_decoder = generative_decoder
        self.video_vae = video_vae
        self.prompt_encoder = prompt_encoder
        self.pad_multiple = pad_multiple
        self.with_color_fix = with_color_fix

        self._to_pil = transforms.ToPILImage()
        self._to_tensor = transforms.ToTensor()

    @classmethod
    def from_pretrained(cls, model_path, checkpoint_dir, dtype="bfloat16",
                        pad_multiple=64, with_color_fix=True, device="cuda"):
        """
        Load all NeFIC components from pretrained checkpoints.

        Args:
            model_path: path to CogVideoX1.5-5B-standard base model
            checkpoint_dir: path to NeFIC training checkpoint
                           (contains anchor_codec.bin + pytorch_lora_weights.safetensors)
            dtype: computation dtype ("bfloat16" or "float16")
            pad_multiple: pad images to this multiple
            with_color_fix: apply AdaIN color correction
            device: target device

        Returns:
            NeFICCodec instance ready for inference
        """
        import os

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        logger.info(f"Loading NeFIC from: {checkpoint_dir}")

        # 1. Load Anchor Codec
        logger.info("Loading Anchor Codec...")
        anchor_codec = AnchorCodec()
        anchor_path = os.path.join(checkpoint_dir, "anchor_codec.bin")
        elic_path = os.path.join(checkpoint_dir, "elicmodel.bin")  # legacy fallback
        if os.path.exists(anchor_path):
            state_dict = torch.load(anchor_path, map_location="cpu")
            anchor_codec.load_state_dict(state_dict, strict=False)
            logger.info(f"  Loaded weights from: {anchor_path}")
        elif os.path.exists(elic_path):
            state_dict = torch.load(elic_path, map_location="cpu")
            anchor_codec.load_state_dict(state_dict, strict=False)
            logger.info(f"  Loaded weights from: {elic_path} (legacy)")
        else:
            raise FileNotFoundError(f"Anchor codec weights not found in: {checkpoint_dir}")
        anchor_codec.to(device).eval()

        # 2. Load Video VAE
        logger.info("Loading Video-VAE...")
        video_vae = VideoVAE.from_pretrained(model_path, dtype=torch_dtype)
        video_vae.to(device)

        # 3. Load Generative Decoder (Transformer + LoRA)
        logger.info("Loading Generative Decoder (CogVideoX + LoRA)...")
        generative_decoder = GenerativeDecoder.from_pretrained(
            model_path, checkpoint_dir, dtype=torch_dtype
        )
        generative_decoder.to(device)

        # 4. Load Prompt Embeddings
        logger.info("Loading Prompt Embeddings...")
        prompt_encoder = PromptEncoder.from_pretrained(
            checkpoint_dir=checkpoint_dir,
            model_path=model_path,
            max_seq_length=generative_decoder.config.max_text_seq_length,
            dtype=torch_dtype,
            device=device,
        )

        codec = cls(
            anchor_codec=anchor_codec,
            generative_decoder=generative_decoder,
            video_vae=video_vae,
            prompt_encoder=prompt_encoder,
            pad_multiple=pad_multiple,
            with_color_fix=with_color_fix,
        )
        logger.info("NeFIC codec loaded successfully")
        return codec

    @torch.no_grad()
    def forward(self, x, use_entropy_coding=False, with_color_fix=None):
        """
        Full encode-decode pipeline.

        Args:
            x: input image [B, 3, H, W] in [0, 1]
            use_entropy_coding: if True, use actual arithmetic coding (precise bpp);
                                if False (default), use STE simulation (fast, estimated bpp)
            with_color_fix: if True, apply AdaIN color correction (matches GT color stats);
                            if False, skip color fix; if None (default), use instance setting

        Returns:
            dict:
                x_hat: [B, 3, H, W] reconstructed image in [0, 1]
                anchor_frame: [B, 3, H, W] anchor reconstruction
                bpp: bits per pixel
        """
        device = x.device
        dtype = next(self.video_vae.parameters()).dtype
        _, _, orig_h, orig_w = x.shape
        num_pixels = orig_h * orig_w

        # Pad to multiple
        x_pad, pads = pad_to_multiple(x, self.pad_multiple)
        _, _, height, width = x_pad.shape

        # === ENCODE ===
        # Step 1: Video-VAE encode → vae_condition (for conditional encoding)
        vae_condition = self.video_vae.encode(x_pad).squeeze(1)  # [B, C, H/8, W/8]

        # Step 2: Anchor Codec
        if use_entropy_coding:
            # Encode → bitstream
            compress_out = self.anchor_codec.compress(x_pad.float(), vae_condition.float())
            # Decode from bitstream → anchor_frame + bypass_latent (decoder side)
            decompress_out = self.anchor_codec.decompress(compress_out["strings"], compress_out["shape"])
            anchor_frame = decompress_out["anchor_frame"].clamp(0., 1.)
            bypass_latent = decompress_out["bypass_latent"]
            # Compute actual bpp from bitstream
            bpp = sum(len(s) * 8 for s_list in compress_out["strings"] for s in (s_list if isinstance(s_list, list) else [s_list])) / num_pixels
        else:
            anchor_out = self.anchor_codec(x_pad.float(), vae_condition.float())
            anchor_frame = anchor_out["anchor_frame"].clamp(0., 1.)
            bypass_latent = anchor_out["bypass_latent"]
            # Estimate bpp from likelihoods
            bpp = 0.0
            for likeli in anchor_out["likelihoods"].values():
                bpp += (torch.log(likeli).sum() / (-math.log(2) * num_pixels)).item()

        # === DECODE ===
        # Step 3: Video-VAE encode anchor frame → anchor_latent (anchor tokens)
        # Note: pass float32 anchor_frame to encode(); the *2-1 normalization is done
        # in float32 inside VideoVAE.encode() before casting to vae dtype (matches training/inference)
        anchor_latent = self.video_vae.encode(anchor_frame)  # [B, 1, C, H/8, W/8]

        # Step 4: Generative Decoder one-step prediction
        prompt_embeds = self.prompt_encoder.encode(device=device, dtype=dtype)
        target_latent = self.generative_decoder.decode(
            anchor_latent=anchor_latent,
            bypass_latent=bypass_latent.to(dtype),
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
        )

        # Step 5: Video-VAE decode → reconstructed image
        x_hat = self.video_vae.decode(target_latent)
        x_hat = x_hat.to(x.dtype)

        # Crop to original size
        x_hat = crop_to_original(x_hat, pads, (orig_h, orig_w))
        anchor_frame_crop = crop_to_original(anchor_frame, pads, (orig_h, orig_w))

        # Optional color correction
        do_color_fix = self.with_color_fix if with_color_fix is None else with_color_fix
        if do_color_fix:
            x_hat = self._apply_color_fix(x_hat, x)

        x_hat = x_hat.clamp(0., 1.)

        return {
            "x_hat": x_hat,
            "anchor_frame": anchor_frame_crop,
            "bpp": bpp,
        }

    def _apply_color_fix(self, x_hat, x_gt):
        """Apply AdaIN color correction per image in batch."""
        B = x_hat.shape[0]
        results = []
        for i in range(B):
            pred_pil = self._to_pil(x_hat[i].cpu())
            gt_pil = self._to_pil(x_gt[i].cpu())
            fixed_pil = adain_color_fix_quant(pred_pil, gt_pil, 16)
            results.append(self._to_tensor(fixed_pil))
        return torch.stack(results, dim=0).to(x_hat.device)

    def compress(self, x):
        """
        Actual entropy encoding (produces real bitstream, encoder side only).

        Args:
            x: input image [B, 3, H, W] in [0, 1]

        Returns:
            dict: strings, shape, bpp, orig_hw, padded_hw
        """
        _, _, orig_h, orig_w = x.shape
        x_pad, pads = pad_to_multiple(x, self.pad_multiple)
        dtype = next(self.video_vae.parameters()).dtype

        vae_condition = self.video_vae.encode(x_pad).squeeze(1)
        anchor_out = self.anchor_codec.compress(x_pad.float(), vae_condition.float())

        num_pixels = orig_h * orig_w
        bpp = sum(len(s) * 8 for s_list in anchor_out["strings"] for s in (s_list if isinstance(s_list, list) else [s_list])) / num_pixels

        return {
            "strings": anchor_out["strings"],
            "shape": anchor_out["shape"],
            "bpp": bpp,
            "orig_hw": (orig_h, orig_w),
            "padded_hw": tuple(x_pad.shape[2:]),
        }

    def decompress(self, compressed):
        """
        Actual entropy decoding + generative reconstruction.

        Args:
            compressed: dict from compress() with strings, shape, etc.

        Returns:
            x_hat: [B, 3, H, W] reconstructed image in [0, 1]
        """
        device = next(self.parameters()).device
        dtype = next(self.video_vae.parameters()).dtype

        # Decompress anchor codec
        anchor_out = self.anchor_codec.decompress(compressed["strings"], compressed["shape"])
        anchor_frame = anchor_out["anchor_frame"].clamp(0., 1.)
        bypass_latent = anchor_out["bypass_latent"]

        # Generative decoding
        height, width = compressed["padded_hw"]
        anchor_latent = self.video_vae.encode(anchor_frame)  # float32 → encode handles dtype
        prompt_embeds = self.prompt_encoder.encode(device=device, dtype=dtype)
        target_latent = self.generative_decoder.decode(
            anchor_latent=anchor_latent,
            bypass_latent=bypass_latent.to(dtype),
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
        )
        x_hat = self.video_vae.decode(target_latent)

        # Crop to original
        orig_h, orig_w = compressed["orig_hw"]
        x_hat = crop_to_original(x_hat, None, (orig_h, orig_w))
        x_hat = x_hat.clamp(0., 1.)
        return x_hat
