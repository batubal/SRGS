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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from utils.gaussian_sr_cameras import (
    generate_camera_params,
    minicams_from_camera_params,
    save_camera_params,
)
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def render_set(model_path, name, method, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, method, "renders")
    gts_path = os.path.join(model_path, name, method, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.image_test[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def _checkpoint_ply(model_path, iteration):
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        return None
    if iteration is None or iteration == -1:
        try:
            iteration = searchForMaxIteration(pc_dir)
        except ValueError:
            return None
    path = os.path.join(pc_dir, "iteration_{}".format(iteration), "point_cloud.ply")
    return path if os.path.isfile(path) else None


def _attach_gt_renders(views, gt_gaussians, pipeline, background):
    """Render GT gaussians at the same cameras (gaussian_sr splat-vs-splat GT)."""
    for view in views:
        gt = render(view, gt_gaussians, pipeline, background)["render"].detach().cpu()
        view.image_test = gt


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, ply_path=None, method=None, gaussian_sr_cameras=False, camera_num_views=8, camera_image_size=256, camera_focal_length=500.0, camera_fit_ratio=0.8):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset, gaussians,
            load_iteration=None if ply_path else iteration,
            shuffle=False, ply_path=ply_path,
            load_cameras=not gaussian_sr_cameras,
        )

        if method is None:
            method = "baseline_lr" if ply_path else "ours_{}".format(scene.loaded_iter)

        if gaussian_sr_cameras:
            background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            ckpt_ply = _checkpoint_ply(dataset.model_path, iteration)
            if ply_path and ckpt_ply:
                gt_gaussians = GaussianModel(dataset.sh_degree)
                gt_gaussians.load_ply(ckpt_ply)
                frame_xyz = gt_gaussians.get_xyz
            else:
                gt_gaussians = gaussians
                frame_xyz = gaussians.get_xyz
                if ply_path:
                    print("No trained checkpoint found; framing cameras on the eval PLY and using it as GT.")

            cam_params = generate_camera_params(
                frame_xyz,
                num_views=camera_num_views,
                image_size=camera_image_size,
                focal_length=camera_focal_length,
                fit_ratio=camera_fit_ratio,
            )
            save_camera_params(
                os.path.join(dataset.model_path, "gaussian_sr_cameras.pt"),
                cam_params,
            )
            views = minicams_from_camera_params(cam_params)
            _attach_gt_renders(views, gt_gaussians, pipeline, background)
            print(
                "gaussian_sr cameras: {} views at {}px, fx={}, black bg, "
                "framed on {}".format(
                    camera_num_views, camera_image_size, camera_focal_length,
                    "trained checkpoint" if gt_gaussians is not gaussians else "eval gaussians",
                )
            )
            if not skip_test:
                render_set(dataset.model_path, "test", method, views, gaussians, pipeline, background)
            return

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", method, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", method, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ply", type=str, default=None,
                        help="Render this Gaussian PLY instead of a trained checkpoint (e.g. SplatFormer splat.ply)")
    parser.add_argument("--method", type=str, default=None,
                        help="metrics.py folder name under test/ (default: ours_<iter> or baseline_lr)")
    parser.add_argument("--gaussian_sr_cameras", action="store_true",
                        help="Render 8 orbit views matching gaussian_sr infer_stage2_diffusion.py (256px, fx=500, black bg)")
    parser.add_argument("--camera_num_views", type=int, default=8)
    parser.add_argument("--camera_image_size", type=int, default=256)
    parser.add_argument("--camera_focal_length", type=float, default=500.0)
    parser.add_argument("--camera_fit_ratio", type=float, default=0.8)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # get_combined_args drops None CLI values, and training cfg_args has no ply/method.
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                ply_path=getattr(args, "ply", None), method=getattr(args, "method", None),
                gaussian_sr_cameras=getattr(args, "gaussian_sr_cameras", False),
                camera_num_views=getattr(args, "camera_num_views", 8),
                camera_image_size=getattr(args, "camera_image_size", 256),
                camera_focal_length=getattr(args, "camera_focal_length", 500.0),
                camera_fit_ratio=getattr(args, "camera_fit_ratio", 0.8))
