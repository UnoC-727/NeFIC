# --------------------------------------------------------------------------------
#   Evaluation script for FID/KID and perceptual metrics.
#   Based on NeuralCompression (https://github.com/facebookresearch/NeuralCompression)
# --------------------------------------------------------------------------------

import sys
import argparse
import tqdm
import pyiqa
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
try:
    from torchmetrics.image import (
        FrechetInceptionDistance,
        KernelInceptionDistance,
        LearnedPerceptualImagePatchSimilarity,
    )
except ImportError:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from neuralcompression.metrics import update_patch_fid

IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

def find_first_image(path_list):
    for p in path_list:
        if p.suffix in IMG_EXTS and p.is_file():
            return p
    return None

def evaluate(recon_dir, gt_dir, ntest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totensor = ToTensor()

    metric_dict = {}

    metric_paired_dict = {}
    recon_dir = Path(recon_dir) if not isinstance(recon_dir, Path) else recon_dir
    assert recon_dir.is_dir()

    gt_path_list = None
    fid_metric = None
    kid_metric = None

    if gt_dir is not None:
        gt_dir = Path(gt_dir) if not isinstance(gt_dir, Path) else gt_dir
        # All GT images
        gt_path_list = sorted([x for x in gt_dir.glob("*") if x.suffix in IMG_EXTS])

        pairs = []
        for gt_p in gt_path_list:
            stem = gt_p.stem  # e.g., "0801"
            # Prefer exact match, then _nefic suffix, then any extension
            primary = recon_dir / f"{stem}{gt_p.suffix}"
            nefic_primary = recon_dir / f"{stem}_nefic.png"
            if primary.exists():
                match = primary
            elif nefic_primary.exists():
                match = nefic_primary
            else:
                # Fallback: match stem or stem_nefic with any extension
                candidates = list(recon_dir.glob(f"{stem}.*")) + list(recon_dir.glob(f"{stem}_nefic.*"))
                match = find_first_image(sorted(candidates))
            if match is not None:
                pairs.append((match, gt_p))

        if ntest is not None:
            pairs = pairs[:ntest]

        # Aligned paired lists
        recon_path_list = [p[0] for p in pairs]
        gt_path_list = [p[1] for p in pairs]

        # Paired metrics
        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr').to(device)
        metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device)
        metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)  # lpips-alexnet
        fid_metric = FrechetInceptionDistance().to(device)
        kid_metric = KernelInceptionDistance().to(device)

    else:
        recon_path_list = sorted(
            [x for x in recon_dir.glob("*_vi_color.*") if x.suffix in IMG_EXTS]
        )
        if ntest is not None:
            recon_path_list = recon_path_list[:ntest]

    if gt_dir is not None:
        print(f'Find {len(recon_path_list)} matched pairs in {recon_dir} <-> {gt_dir}')
    else:
        print(f'Find {len(recon_path_list)} images in {recon_dir} ')

    result = {}
    for i in tqdm.tqdm(range(len(recon_path_list))):
        recon_path = str(recon_path_list[i])
        gt_path = str(gt_path_list[i]) if gt_path_list is not None else None

        with open(recon_path, "rb") as f:
            image_recon = Image.open(f).convert("RGB")
        recon_tensor = totensor(image_recon).unsqueeze(0).to(device)

        # No-reference metrics
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                value = metric(recon_tensor).item()
                result[key] = result.get(key, 0) + value

        # Paired metrics
        if gt_path is not None:
            with open(gt_path, "rb") as f:
                image_gt = Image.open(f).convert("RGB")
            gt_tensor = totensor(image_gt).unsqueeze(0).to(device)

            if fid_metric is not None and kid_metric is not None:
                update_patch_fid(gt_tensor, recon_tensor, fid_metric=fid_metric, kid_metric=kid_metric)

            for key, metric in metric_paired_dict.items():
                value = metric(recon_tensor, gt_tensor).item()
                result[key] = result.get(key, 0) + value

    if gt_dir is not None and len(recon_path_list) > 50 and fid_metric is not None:
        result['fid'] = float(fid_metric.compute())
        kid_tuple = kid_metric.compute()
        result['kid_mean'], result['kid_std'] = float(kid_tuple[0]), float(kid_tuple[1])

    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        elif key == 'kid_mean' or key == 'kid_std':
            print(f"{key}: {res:.7f}")
            print_results.append(f"{key}: {res:.7f}")
        else:
            denom = max(len(recon_path_list), 1)
            print(f"{key}: {res/denom:.5f}")
            print_results.append(f"{key}: {res/denom:.5f}")
    return print_results


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example evaluation script.")
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print_results = evaluate(args.recon_dir, args.gt_dir, args.ntest)

if __name__ == "__main__":
    main(sys.argv[1:])
