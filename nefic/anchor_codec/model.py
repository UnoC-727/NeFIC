"""
AnchorCodec: VAE-conditioned learned image compression with bypass refinement.

This is the core compression model of NeFIC. It produces:
  - anchor_frame: a compact reconstruction preserving geometry and semantics
  - bypass_latent (z_bypass): initialization for one-step video diffusion generation
  - likelihoods: for bitrate (bpp) computation
"""

import time
import math

import torch
import torch.nn as nn

from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder

from .encoder import AnchorEncoder
from .decoder import AnchorDecoder
from .layers import ResidualTokenProjectionBlock
from .entropy import (
    HyperAnalysis,
    HyperSynthesis,
    EntropyParameters,
    LatentResidualPrediction,
    LocalContext,
    ChannelContext,
    LinearGlobalIntraContext,
    LinearGlobalInterContext,
)
from .layers import (
    ckbd_split,
    ckbd_merge,
    ckbd_anchor,
    ckbd_nonanchor,
    compress_anchor,
    compress_nonanchor,
    decompress_anchor,
    decompress_nonanchor,
)


def ste_round(x):
    """Straight-through estimator rounding."""
    return torch.round(x) - x.detach() + x


def get_scale_table(min_val=0.11, max_val=256, levels=64):
    """Generate scale table for Gaussian conditional."""
    return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))


def update_registered_buffers(module, module_name, buffer_names, state_dict,
                              policy="resize_if_empty", dtype=torch.int):
    """Update registered buffers in a module from state_dict."""
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names:
        key = f"{module_name}.{buffer_name}"
        if key not in state_dict:
            continue
        new_size = state_dict[key].size()
        registered_buf = next((b for n, b in module.named_buffers() if n == buffer_name), None)
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)


class AnchorCodecConfig:
    """Default configuration for AnchorCodec."""
    N = 192
    M = 320
    slice_num = 10
    context_window = 5


class AnchorCodec(CompressionModel):
    """
    Anchor Codec: VAE-conditioned learned image compression.

    Encodes images conditioned on Video-VAE latents to align with the
    video generation domain, and produces a bypass latent for one-step
    video diffusion initialization.

    Architecture:
        - AnchorEncoder: analysis transform with conditional encoding
        - AnchorDecoder: synthesis transform outputting anchor frame + mid features
        - BypassRefinement: maps mid features to VDM-compatible bypass latent
        - Checkerboard entropy model with local/global context
    """

    def __init__(self, config=None):
        config = config or AnchorCodecConfig()
        N = config.N
        M = config.M
        super().__init__(N)

        context_window = config.context_window
        slice_num = config.slice_num
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.N = N
        self.M = M
        self.context_window = context_window
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        # Core transforms
        self.g_a = AnchorEncoder(N=N, M=M)
        self.g_s = AnchorDecoder(N=N, M=M)

        # Bypass Refinement (parameters at top level to match checkpoint key layout)
        # Checkpoint stores: project_in.*, information_transformer_layes.*, spatial_ch_projs.*
        num_projection_layers = 3
        projection_channels = 320
        out_channels = 16
        self.project_in = nn.Linear(N, projection_channels)
        self.information_transformer_layes = nn.Sequential(
            *[ResidualTokenProjectionBlock(projection_channels)
              for _ in range(num_projection_layers)]
        )
        self.spatial_ch_projs = nn.Linear(projection_channels, out_channels)

        # Hyper-prior
        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        # Gaussian conditional
        self.gaussian_conditional = GaussianConditional(None)

        # Context models
        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch) for _ in range(slice_num)
        )
        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        self.global_inter_context = nn.ModuleList(
            LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        # Entropy parameter estimation
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )

        # Latent residual prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )

    def _update_resolutions(self, H, W):
        """Update resolution-dependent masks for local context."""
        for i in range(len(self.global_intra_context)):
            if i == 0:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=None)
            else:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=self.local_context[0].attn_mask)

    def _bypass_refinement(self, mid_feat):
        """
        Bypass Refinement: maps anchor decoder mid features to VDM latent space.
        Parameters are at top-level to match checkpoint key layout.
        """
        B, C, H, W = mid_feat.shape
        feat = mid_feat.view(B, C, H * W).transpose(1, 2)
        feat = self.project_in(feat)
        feat = self.information_transformer_layes(feat)
        feat = self.spatial_ch_projs(feat).transpose(1, 2).view(B, -1, H, W)
        return feat.unsqueeze(1)  # [B, 1, 16, H, W]

    def forward(self, x, vae_condition):
        """
        STE-based forward pass (no actual entropy coding).

        Args:
            x: input image [B, 3, H, W] in [0, 1]
            vae_condition: Video-VAE latent [B, 16, H/8, W/8]

        Returns:
            dict:
                anchor_frame: [B, 3, H, W] reconstructed anchor
                bypass_latent: [B, 1, 16, H/4, W/4] for VDM initialization
                likelihoods: dict of y and z likelihoods
        """
        self._update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x, vae_condition)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihoods = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
            else:
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)

            y_hat_slices.append(y_hat_slice)
            y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)

        anchor_frame, mid_feat = self.g_s(y_hat)
        bypass_latent = self._bypass_refinement(mid_feat)

        return {
            "anchor_frame": anchor_frame,
            "bypass_latent": bypass_latent,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods},
        }

    def compress(self, x, vae_condition):
        """
        Actual arithmetic encoding → bitstream (encoder side only).

        Args:
            x: input image [B, 3, H, W] in [0, 1]
            vae_condition: Video-VAE latent [B, 16, H/8, W/8]

        Returns:
            dict: strings, shape, cost_time
        """
        torch.cuda.synchronize()
        start_time = time.time()

        self._update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x, vae_condition)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
            else:
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.cuda.synchronize()
        cost_time = time.time() - start_time

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time,
        }

    def decompress(self, strings, shape):
        """
        Actual arithmetic decoding from bitstream.

        Args:
            strings: [y_strings, z_strings]
            shape: spatial shape of z

        Returns:
            dict: anchor_frame, bypass_latent, cost_time
        """
        torch.cuda.synchronize()
        start_time = time.time()

        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self._update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)

        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
            else:
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        anchor_frame, mid_feat = self.g_s(y_hat)
        bypass_latent = self._bypass_refinement(mid_feat)

        torch.cuda.synchronize()
        cost_time = time.time() - start_time

        return {
            "anchor_frame": anchor_frame,
            "bypass_latent": bypass_latent,
            "cost_time": cost_time,
        }

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=False)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
