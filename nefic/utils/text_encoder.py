"""Text/Prompt encoder for NeFIC's generation guidance."""

import os
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PromptEncoder(nn.Module):
    """
    Manages prompt embeddings for NeFIC's generation guidance.

    NeFIC uses a universal prompt to guide the video diffusion model.
    The prompt embedding is fixed for all images, so it can be:
      1. Loaded from a pre-saved .pt file (fast, no T5 needed)
      2. Computed on-the-fly via T5 encoder (fallback for old checkpoints)
    """

    # Default prompt (aligned with training)
    DEFAULT_PROMPT = (
        "Generate high-resolution, high-quality images with realistic textures. "
        "The image has the highly compressed map."
    )
    DEFAULT_INSTANCE_TOKEN = "compressed"
    EMBED_FILENAME = "prompt_embeds.pt"

    def __init__(self, cached_embeds=None):
        super().__init__()
        self._cached_embeds = cached_embeds

    @classmethod
    def from_pretrained(cls, checkpoint_dir, model_path=None,
                        max_seq_length=None, dtype=torch.bfloat16, device="cuda"):
        """
        Load prompt embeddings from assets/prompt_embeds.pt.

        Args:
            checkpoint_dir: NeFIC checkpoint directory (unused, kept for API compatibility)
            model_path: path to CogVideoX base model (needed only for fallback)
            max_seq_length: max token sequence length (from transformer config)
            dtype: target dtype for embeddings
            device: target device
        """
        # Load from assets/prompt_embeds.pt relative to package root
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        embed_path = os.path.join(pkg_root, "assets", cls.EMBED_FILENAME)

        if os.path.exists(embed_path):
            logger.info(f"  Loading cached prompt embeddings from: {embed_path}")
            cached_embeds = torch.load(embed_path, map_location=device)
            cached_embeds = cached_embeds.to(dtype=dtype, device=device)
            return cls(cached_embeds=cached_embeds)

        # Fallback: compute via T5 encoder
        if model_path is None:
            raise FileNotFoundError(
                f"Prompt embeddings not found at {embed_path} and no model_path "
                f"provided for T5 fallback. Run `python -m nefic.utils.text_encoder "
                f"--model_path <path> --output {embed_path}` to generate it."
            )

        logger.info("  prompt_embeds.pt not found, computing via T5 (fallback)...")
        from transformers import AutoTokenizer, T5EncoderModel

        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        ).to(device, dtype=dtype)
        text_encoder.eval()

        max_seq_length = max_seq_length or 226  # CogVideoX-1.5 default

        text_inputs = tokenizer(
            [cls.DEFAULT_PROMPT],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # Save for future use
        torch.save(prompt_embeds.cpu(), embed_path)
        logger.info(f"  Saved prompt embeddings to: {embed_path}")

        # Free T5 memory
        del text_encoder, tokenizer
        torch.cuda.empty_cache()

        return cls(cached_embeds=prompt_embeds)

    def encode(self, device="cuda", dtype=torch.bfloat16):
        """
        Return cached prompt embeddings.

        Returns:
            prompt_embeds: [1, seq_len, D]
        """
        if self._cached_embeds is None:
            raise RuntimeError("PromptEncoder has no cached embeddings. "
                               "Use PromptEncoder.from_pretrained() to load.")
        return self._cached_embeds.to(device=device, dtype=dtype)

    def free_memory(self):
        """No-op (kept for API compatibility). T5 is never held in memory."""
        pass


# ============ CLI: Generate prompt_embeds.pt ============

if __name__ == "__main__":
    """
    Utility script to pre-compute and save prompt embeddings.

    Usage:
        python -m nefic.utils.text_encoder \
            --model_path /path/to/CogVideoX1.5-5B-standard \
            --output /path/to/checkpoint/prompt_embeds.pt \
            [--max_seq_length 226] [--dtype bfloat16]
    """
    import argparse

    parser = argparse.ArgumentParser("Generate prompt_embeds.pt")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to CogVideoX base model")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for prompt_embeds.pt")
    parser.add_argument("--max_seq_length", type=int, default=226,
                        help="Max text sequence length (default: 226 for CogVideoX-1.5)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer, T5EncoderModel

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading T5 from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_path, subfolder="text_encoder"
    ).to(device, dtype=torch_dtype)
    text_encoder.eval()

    text_inputs = tokenizer(
        [PromptEncoder.DEFAULT_PROMPT],
        padding="max_length",
        max_length=args.max_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=torch_dtype, device="cpu")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(prompt_embeds, args.output)
    logger.info(f"Saved: {args.output} (shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype})")
