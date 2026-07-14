#!/usr/bin/env python3
"""
NeFIC Inference Test Script.

Evaluates NeFIC codec on image datasets (Kodak, CLIC, DIV2K).
Reports PSNR, MS-SSIM, LPIPS, DISTS, and BPP.

Usage:
    python test.py \
        --model_path /path/to/CogVideoX1.5-5B-standard \
        --checkpoint_dir /path/to/nefic_checkpoint \
        --input_dir /path/to/test_images \
        --output_dir ./results
"""

import os
import sys
import time
import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path

import pyiqa

# Add parent directory for nefic package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nefic import NeFICCodec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TestImageFolder(Dataset):
    """Simple dataset for test images."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"Test data dir not found: {self.root}")
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.image_paths = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in img_exts
        ])
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root}")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return {"pixel_values": tensor, "path": str(path)}


def parse_args():
    parser = argparse.ArgumentParser("NeFIC Inference")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="./nefic_results",
                        help="Output directory")
    parser.add_argument("--model_path", type=str,
                        required=True,
                        help="Path to CogVideoX base model")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to NeFIC checkpoint directory")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--pad_multiple", type=int, default=64)
    parser.add_argument("--with_color_fix", action="store_true", default=True)
    parser.add_argument("--no_color_fix", action="store_true", default=False)
    parser.add_argument("--use_entropy_coding", action="store_true", default=False,
                        help="Use actual arithmetic coding (slower, precise bpp)")
    parser.add_argument("--save_intermediate", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with_color_fix = args.with_color_fix and not args.no_color_fix

    logger.info(f"Device: {device}, dtype: {args.dtype}")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}")
    logger.info(f"Color fix: {with_color_fix}")
    logger.info(f"Entropy coding: {args.use_entropy_coding}")

    # Load NeFIC codec
    codec = NeFICCodec.from_pretrained(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        dtype=args.dtype,
        pad_multiple=args.pad_multiple,
        with_color_fix=with_color_fix,
        device=device,
    )

    # Dataset
    test_dataset = TestImageFolder(args.input_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"Found {len(test_dataset)} images")

    # Metrics
    metric_psnr = pyiqa.create_metric("psnr").to(device)
    metric_msssim = pyiqa.create_metric("ms_ssim").to(device)
    metric_lpips = pyiqa.create_metric("lpips", as_loss=False).to(device)
    metric_dists = pyiqa.create_metric("dists").to(device)

    # Accumulators
    results = {"psnr": 0., "ms_ssim": 0., "lpips": 0., "dists": 0., "bpp": 0.}
    count = 0
    t0 = time.time()

    # Inference
    with torch.no_grad():
        for idx, batch in enumerate(test_loader, 1):
            x_gt = batch["pixel_values"].to(device)
            file_path = batch["path"][0]
            img_name = os.path.splitext(os.path.basename(file_path))[0]
            _, _, h, w = x_gt.shape

            logger.info(f"[{idx}/{len(test_dataset)}] {img_name} ({w}x{h})")

            # Forward pass
            out = codec(x_gt, use_entropy_coding=args.use_entropy_coding)
            x_hat = out["x_hat"]
            bpp = out["bpp"]

            # Save outputs
            save_image(x_hat, os.path.join(args.output_dir, f"{img_name}_nefic.png"))
            if args.save_intermediate:
                save_image(out["anchor_frame"], os.path.join(args.output_dir, f"{img_name}_anchor.png"))

            # Compute metrics
            results["psnr"] += metric_psnr(x_hat, x_gt).item()
            results["ms_ssim"] += metric_msssim(x_hat, x_gt).item()
            results["lpips"] += metric_lpips(x_hat, x_gt).item()
            results["dists"] += metric_dists(x_hat, x_gt).item()
            results["bpp"] += bpp
            count += 1

    # Print results
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"NeFIC Results ({count} images, {elapsed:.1f}s total, {elapsed/count:.1f}s/image)")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print("=" * 60)
    print(f"  PSNR:    {results['psnr']/count:.3f} dB")
    print(f"  MS-SSIM: {results['ms_ssim']/count:.6f}")
    print(f"  LPIPS:   {results['lpips']/count:.6f}")
    print(f"  DISTS:   {results['dists']/count:.6f}")
    print(f"  BPP:     {results['bpp']/count:.6f}")
    print("=" * 60)

    # Save to file
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Checkpoint: {args.checkpoint_dir}\n")
        f.write(f"Input: {args.input_dir}\n")
        f.write(f"Images: {count}\n")
        f.write(f"Time: {elapsed:.1f}s ({elapsed/count:.1f}s/image)\n")
        f.write(f"Entropy coding: {args.use_entropy_coding}\n\n")
        f.write(f"PSNR: {results['psnr']/count:.3f} dB\n")
        f.write(f"MS-SSIM: {results['ms_ssim']/count:.6f}\n")
        f.write(f"LPIPS: {results['lpips']/count:.6f}\n")
        f.write(f"DISTS: {results['dists']/count:.6f}\n")
        f.write(f"BPP: {results['bpp']/count:.6f}\n")


if __name__ == "__main__":
    main()
