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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from utils.image_utils import psnr_joint, psnr_joint_from_list, ssim_skimage
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def _init_lpips(device):
    """VGG LPIPS in the same [-1, 1] convention as gaussian_sr."""
    try:
        import lpips as official_lpips
        fn = official_lpips.LPIPS(net="vgg").to(device)
    except ImportError:
        from lpipsPyTorch.modules.lpips import LPIPS
        fn = LPIPS(net_type="vgg").to(device)
    fn.eval()
    return fn


def _lpips_batch(lpips_fn, preds, gts):
    """Mean LPIPS over views; inputs mapped from [0, 1] to [-1, 1]."""
    vals = []
    with torch.no_grad():
        for pred, gt in zip(preds, gts):
            pred_lp = pred.float().clamp(0, 1) * 2.0 - 1.0
            gt_lp = gt.float().clamp(0, 1) * 2.0 - 1.0
            vals.append(lpips_fn(pred_lp, gt_lp).mean().item())
    return vals


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    print("")

    device = torch.device("cuda:0")
    lpips_fn = _init_lpips(device)

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in sorted(os.listdir(test_dir)):
            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            if not renders_dir.is_dir() or not gt_dir.is_dir():
                continue

            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            renders, gts, image_names = readImages(renders_dir, gt_dir)

            per_view_psnrs = []
            per_view_ssims = []
            per_view_lpips = []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                per_view_psnrs.append(psnr_joint(renders[idx], gts[idx]))
                per_view_ssims.append(ssim_skimage(renders[idx], gts[idx]))
                per_view_lpips.extend(_lpips_batch(lpips_fn, [renders[idx]], [gts[idx]]))

            # Joint MSE over all views — same logged PSNR as gaussian_sr.
            psnr_val = psnr_joint_from_list(renders, gts)
            ssim_val = float(sum(per_view_ssims) / max(len(per_view_ssims), 1))
            lpips_val = float(sum(per_view_lpips) / max(len(per_view_lpips), 1))
            psnr_per_view_mean = float(sum(per_view_psnrs) / max(len(per_view_psnrs), 1))

            print("  SSIM : {:>12.7f}".format(ssim_val))
            print("  PSNR : {:>12.7f}  (joint MSE over all views)".format(psnr_val))
            print("  PSNR per-view mean : {:>12.7f}".format(psnr_per_view_mean))
            print("  LPIPS: {:>12.7f}".format(lpips_val))
            print("")

            full_dict[scene_dir][method].update({
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "PSNR_per_view_mean": psnr_per_view_mean,
                "LPIPS": lpips_val,
            })
            per_view_dict[scene_dir][method].update({
                "SSIM": {name: s for s, name in zip(per_view_ssims, image_names)},
                "PSNR": {name: p for p, name in zip(per_view_psnrs, image_names)},
                "LPIPS": {name: lp for lp, name in zip(per_view_lpips, image_names)},
            })

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

        methods = full_dict[scene_dir]
        if len(methods) > 1:
            print("Summary")
            print("  {:<20} {:>12} {:>12} {:>12}".format("Method", "SSIM", "PSNR", "LPIPS"))
            for name in sorted(methods):
                m = methods[name]
                print("  {:<20} {:>12.7f} {:>12.7f} {:>12.7f}".format(
                    name, m["SSIM"], m["PSNR"], m["LPIPS"]))
            print("")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
