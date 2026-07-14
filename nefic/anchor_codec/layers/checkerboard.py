"""Checkerboard context utilities for entropy coding."""

import torch
from compressai.entropy_models import GaussianConditional, EntropyModel


def ckbd_split(y):
    """Split y into anchor and non-anchor parts (checkerboard pattern)."""
    anchor = ckbd_anchor(y)
    nonanchor = ckbd_nonanchor(y)
    return anchor, nonanchor


def ckbd_merge(anchor, nonanchor):
    """Merge anchor and non-anchor parts."""
    return anchor + nonanchor


def ckbd_anchor(y):
    """Extract anchor positions from checkerboard pattern."""
    anchor = torch.zeros_like(y).to(y.device)
    anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
    return anchor


def ckbd_nonanchor(y):
    """Extract non-anchor positions from checkerboard pattern."""
    nonanchor = torch.zeros_like(y).to(y.device)
    nonanchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
    return nonanchor


def ckbd_anchor_sequeeze(y):
    """Squeeze anchor positions to compact representation."""
    B, C, H, W = y.shape
    anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
    return anchor


def ckbd_nonanchor_sequeeze(y):
    """Squeeze non-anchor positions to compact representation."""
    B, C, H, W = y.shape
    nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
    return nonanchor


def ckbd_anchor_unsequeeze(anchor):
    """Unsqueeze anchor positions back to full spatial grid."""
    B, C, H, W = anchor.shape
    y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
    y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
    y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
    return y_anchor


def ckbd_nonanchor_unsequeeze(nonanchor):
    """Unsqueeze non-anchor positions back to full spatial grid."""
    B, C, H, W = nonanchor.shape
    y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
    y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
    y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
    return y_nonanchor


# === Arithmetic coding helpers ===

def compress_anchor(gaussian_conditional, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
    """Compress anchor symbols using arithmetic coding."""
    anchor_squeeze = ckbd_anchor_sequeeze(anchor)
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze)
    anchor_hat = gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
    symbols_list.extend(anchor_hat.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
    return anchor_hat


def compress_nonanchor(gaussian_conditional, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
    """Compress non-anchor symbols using arithmetic coding."""
    nonanchor_squeeze = ckbd_nonanchor_sequeeze(nonanchor)
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
    nonanchor_hat = gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
    symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
    return nonanchor_hat


def decompress_anchor(gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
    """Decompress anchor symbols from bitstream."""
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze)
    anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) + means_anchor_squeeze
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat)
    return anchor_hat


def decompress_nonanchor(gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
    """Decompress non-anchor symbols from bitstream."""
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
    nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) + means_nonanchor_squeeze
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat)
    return nonanchor_hat
