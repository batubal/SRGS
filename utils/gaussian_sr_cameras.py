"""Camera generation matching gaussian_sr ``Renderer.generate_camera_params``."""

import math
import os
import torch
from scene.cameras import MiniCam
from utils.graphics_utils import focal2fov, getProjectionMatrix


def generate_camera_params(
    points: torch.Tensor,
    num_views: int = 8,
    image_size: int = 256,
    focal_length: float = 500.0,
    fit_ratio: float = 0.8,
) -> dict:
    """Create the same synthetic orbit as gaussian_sr (Z-up, +Z forward, black bg)."""
    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]
    center = (min_bounds + max_bounds) / 2
    radius = torch.norm(max_bounds - min_bounds) / 2

    proj_radius_target = max(float(fit_ratio * image_size) * 0.5, 1.0)
    fx_px = float(focal_length)
    if float(radius) > 0:
        distance_fit = fx_px * float(radius) / proj_radius_target
    else:
        distance_fit = 1.0
    camera_distance = max(2.0 * float(radius), distance_fit) * 1.05

    cameras = {
        "camera_to_worlds": [],
        "fx": torch.tensor(focal_length, dtype=center.dtype, device=center.device),
        "fy": torch.tensor(focal_length, dtype=center.dtype, device=center.device),
        "cx": torch.tensor(image_size / 2, dtype=center.dtype, device=center.device),
        "cy": torch.tensor(image_size / 2, dtype=center.dtype, device=center.device),
        "width": torch.tensor(image_size, dtype=center.dtype, device=center.device),
        "height": torch.tensor(image_size, dtype=center.dtype, device=center.device),
        "background_color": torch.tensor([0.0, 0.0, 0.0], dtype=center.dtype, device=center.device),
    }

    for i in range(num_views):
        azimuth = 2 * math.pi * i / num_views
        elevation = math.pi / 4

        x = center[0] + camera_distance * math.cos(azimuth) * math.sin(elevation)
        y = center[1] + camera_distance * math.sin(azimuth) * math.sin(elevation)
        z = center[2] + camera_distance * math.cos(elevation)

        camera_pos = torch.tensor([x, y, z], dtype=center.dtype, device=center.device)

        forward = center - camera_pos
        forward = forward / torch.norm(forward)

        up = torch.tensor([0, 0, 1], dtype=center.dtype, device=center.device)
        right = torch.linalg.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.linalg.cross(right, forward)

        c2w = torch.eye(4, dtype=center.dtype, device=center.device)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = camera_pos
        cameras["camera_to_worlds"].append(c2w)

    cameras["camera_to_worlds"] = torch.stack(cameras["camera_to_worlds"])
    return cameras


def camera_params_to_cpu(cam_params: dict) -> dict:
    out = {}
    for key, value in cam_params.items():
        out[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def camera_params_to_device(cam_params: dict, device) -> dict:
    out = {}
    for key, value in cam_params.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def save_camera_params(path: str, cam_params: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(camera_params_to_cpu(cam_params), path)


def load_camera_params(path: str, device) -> dict:
    return camera_params_to_device(torch.load(path, map_location="cpu"), device)


def minicams_from_camera_params(cam_params: dict) -> list:
    """Convert gaussian_sr c2w / pinhole K into 3DGS MiniCam (same w2c as gsplat)."""
    c2ws = cam_params["camera_to_worlds"]
    fx = float(cam_params["fx"])
    fy = float(cam_params["fy"])
    width = int(cam_params["width"])
    height = int(cam_params["height"])
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    znear, zfar = 0.01, 100.0

    cams = []
    for i in range(c2ws.shape[0]):
        w2c = torch.inverse(c2ws[i].float())
        world_view_transform = w2c.transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
        full_proj = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
        cam = MiniCam(
            width, height, fovy, fovx, znear, zfar,
            world_view_transform, full_proj,
        )
        cams.append(cam)
    return cams
