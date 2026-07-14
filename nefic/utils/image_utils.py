"""Image processing utilities."""

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor


def pad_to_multiple(x: torch.Tensor, multiple: int = 64):
    """
    Pad tensor to the nearest multiple of `multiple`.

    Args:
        x: [B, C, H, W] input tensor
        multiple: pad to this multiple

    Returns:
        x_pad: padded tensor
        pads: (left, right, top, bottom) padding values
    """
    _, _, h, w = x.shape
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    pad_bottom = new_h - h
    pad_right = new_w - w
    x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
    return x_pad, (0, pad_right, 0, pad_bottom)


def crop_to_original(x: torch.Tensor, pads, orig_hw):
    """
    Crop padded tensor back to original size.

    Args:
        x: [B, C, H, W] padded tensor
        pads: padding values (unused, kept for API compatibility)
        orig_hw: (H, W) original dimensions

    Returns:
        Cropped tensor [B, C, orig_H, orig_W]
    """
    h, w = orig_hw
    return x[:, :, :h, :w]


class AsyncImageSaver:
    """Non-blocking image saver using a background thread pool."""

    def __init__(self, max_workers=2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []

    def submit_tensor(self, gpu_tensor, path):
        """Save a GPU tensor as an image file (async)."""
        cpu_tensor = gpu_tensor.detach().cpu().clone()
        future = self._pool.submit(save_image, cpu_tensor, path)
        self._futures.append(future)

    def submit_pil(self, pil_img, path):
        """Save a PIL image to file (async)."""
        future = self._pool.submit(pil_img.save, path)
        self._futures.append(future)

    def wait_all(self):
        """Wait for all pending saves to complete."""
        errors = []
        for f in self._futures:
            try:
                f.result()
            except Exception as e:
                errors.append(e)
        self._futures.clear()
        return errors

    def shutdown(self):
        """Wait for all saves and shut down the thread pool."""
        self.wait_all()
        self._pool.shutdown(wait=True)
