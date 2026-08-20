#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import numpy as np
import torch
import torch.nn.functional as F


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def psnr_joint(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR matching gaussian_sr ``infer_stage2_diffusion._psnr``.

    Single MSE over every pixel (and view, if batched), then ``-10 log10(mse)``.
    """
    mse_val = F.mse_loss(pred.float().clamp(0, 1), gt.float().clamp(0, 1)).item()
    return float("inf") if mse_val < 1e-10 else -10.0 * math.log10(mse_val)


def psnr_joint_from_list(preds, gts) -> float:
    """Joint PSNR over a list of images without stacking them on GPU."""
    sse = 0.0
    numel = 0
    for pred, gt in zip(preds, gts):
        p = pred.float().clamp(0, 1)
        g = gt.float().clamp(0, 1)
        sse += torch.square(p - g).sum().item()
        numel += p.numel()
    mse_val = sse / max(numel, 1)
    return float("inf") if mse_val < 1e-10 else -10.0 * math.log10(mse_val)


def _to_hwc_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert [3,H,W] or [1,3,H,W] in [0, 1] to [H,W,3] float32."""
    t = img.detach().float().clamp(0, 1).cpu()
    if t.dim() == 4:
        t = t[0]
    return t.permute(1, 2, 0).numpy().astype(np.float32)


def ssim_skimage(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM matching gaussian_sr ``infer_stage2_diffusion._ssim`` (one view)."""
    return ssim_skimage_batch([pred], [gt])


def ssim_skimage_batch(preds, gts) -> float:
    """Mean per-view skimage SSIM, matching gaussian_sr."""
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        print(f"WARNING: SSIM computation failed: {exc}")
        return float("nan")

    scores = []
    for pred, gt in zip(preds, gts):
        p = _to_hwc_numpy(pred)
        g = _to_hwc_numpy(gt)
        try:
            h, w = p.shape[:2]
            win = min(7, h, w)
            win = win if win % 2 == 1 else win - 1
            s = structural_similarity(
                p, g, data_range=1.0, channel_axis=-1, win_size=max(1, win)
            )
        except TypeError:
            s = structural_similarity(p, g, data_range=1.0, multichannel=True)
        scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")
