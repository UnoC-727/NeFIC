"""
Generative Decoder (G): One-step video diffusion model for image reconstruction.

Uses CogVideoX-1.5 transformer with LoRA adapters to perform one-step
next-frame prediction, converting the anchor frame into a high-fidelity reconstruction.
"""

import os
import torch
import torch.nn as nn

from diffusers import CogVideoXDPMScheduler, CogVideoXTransformer3DModel

from .rope import prepare_rotary_positional_embeddings


class GenerativeDecoder(nn.Module):
    """
    One-step Video Diffusion Decoder.

    Performs single-step next-frame prediction using CogVideoX transformer
    with LoRA fine-tuning. The input consists of:
      - anchor_latent: VAE-encoded anchor frame tokens (conditioning)
      - bypass_latent: semantic bypass from anchor codec (initialization)
      - prompt_embeds: text embeddings for generation guidance

    The model operates at a fixed timestep t* = T/2 and uses v-prediction
    to reconstruct the clean target latent in one step.
    """

    def __init__(self, transformer, scheduler):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

        # Cache model config
        model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
        self.config = model_config

    @classmethod
    def from_pretrained(cls, model_path, checkpoint_dir, dtype=torch.bfloat16):
        """
        Load generative decoder from pretrained CogVideoX + LoRA weights.

        Args:
            model_path: path to CogVideoX1.5-5B-standard base model
            checkpoint_dir: path to NeFIC checkpoint (contains pytorch_lora_weights.safetensors)
            dtype: computation dtype
        """
        # Load base transformer
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype,
        )

        # Load LoRA weights
        lora_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
        if os.path.exists(lora_path):
            from peft import LoraConfig, set_peft_model_state_dict
            from diffusers.utils import convert_unet_state_dict_to_peft
            from diffusers import CogVideoXPipeline

            lora_config = LoraConfig(
                r=256,
                lora_alpha=256,
                init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "norm1.linear", "norm2.linear"],
            )
            transformer.add_adapter(lora_config)

            lora_state_dict = CogVideoXPipeline.lora_state_dict(
                checkpoint_dir, weight_name="pytorch_lora_weights.safetensors"
            )
            transformer_state_dict = {
                k.replace("transformer.", ""): v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

        transformer.eval()

        # Load scheduler
        scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return cls(transformer=transformer, scheduler=scheduler)

    def decode(self, anchor_latent, bypass_latent, prompt_embeds, height, width):
        """
        One-step generative decoding.

        Args:
            anchor_latent: [B, 1, C, H, W] - VAE-encoded anchor frame tokens
            bypass_latent: [B, 1, C, H, W] - bypass latent from anchor codec
            prompt_embeds: [B, seq_len, D] - text embeddings
            height, width: pixel-space spatial dimensions (for RoPE computation)

        Returns:
            target_latent: [B, 1, C, H, W] - predicted clean latent for VAE decoding
        """
        device = anchor_latent.device
        B = anchor_latent.shape[0]
        model_config = self.config
        vae_scale_factor_spatial = 8  # CogVideoX VAE

        # Prepare RoPE (num_frames=2, then doubled for the 4-frame input)
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height,
                width=width,
                num_frames=2,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                patch_size=model_config.patch_size,
                patch_size_t=model_config.patch_size_t if model_config.patch_size_t is not None else 1,
                attention_head_dim=model_config.attention_head_dim,
                device=device,
            )
            if model_config.use_rotary_positional_embeddings
            else None
        )
        if image_rotary_emb is not None:
            image_rotary_emb = [torch.cat([emb, emb], dim=0) for emb in image_rotary_emb]

        # Construct noisy_model_input: cat(anchor_repeat2, bypass_repeat2) → [B, 4, C, H, W]
        bypass_input = bypass_latent.repeat(1, 2, 1, 1, 1)   # [B, 2, C, H, W]
        cond_input = anchor_latent.repeat(1, 2, 1, 1, 1)     # [B, 2, C, H, W]
        noisy_model_input = torch.cat((cond_input, bypass_input), dim=1)  # [B, 4, C, H, W]

        # Fixed timestep t* = T/2 (one-step generation schedule)
        target_timestep = self.scheduler.config.num_train_timesteps // 2
        timesteps = torch.full((B,), target_timestep, dtype=torch.long, device=device)

        # Transformer forward
        noisy_model_input = noisy_model_input.to(prompt_embeds.dtype)
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        # Velocity prediction → clean latent
        model_pred = self.scheduler.get_velocity(model_output, noisy_model_input, timesteps)
        target_latent = model_pred[:, -2:]  # Take last 2 frames
        target_latent = target_latent[:, -1:]  # Use last frame as final prediction

        return target_latent
