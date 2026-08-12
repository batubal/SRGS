#!/usr/bin/env python3
"""
Convert SplatFormer / Nerfstudio `nerf_dataset` folders into SRGS Blender format.

Input layout (as in lr_data_splatformer):
  <root>/<synset>/.work/<scene_id>/nerf_dataset/
      transforms_{train,test,val}.json
      train/r_*.png
  <root>/<synset>/.work/<scene_id>/export/splat.ply   (optional)
  <root>/<synset>/<scene_id>.ply                      (optional)

Output layout (SRGS Blender / NeRF synthetic):
  <out>/<scene_id>/
      transforms_train.json
      transforms_test.json
      train/
      points3d.ply

Usage:
  python scripts/convert_splatformer_lr.py \
      --src lr_data_splatformer \
      --out data/splatformer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SH_C0 = 0.28209479177387814


def discover_scenes(src: Path) -> List[Tuple[str, Path]]:
    scenes = []
    for nerf_dir in sorted(src.rglob("nerf_dataset")):
        if not (nerf_dir / "transforms_train.json").is_file():
            continue
        scene_id = nerf_dir.parent.name
        if scene_id == ".work":
            continue
        scenes.append((scene_id, nerf_dir))
    return scenes


def find_splat_ply(nerf_dir: Path, scene_id: str) -> Optional[Path]:
    candidates = [
        nerf_dir.parent / "export" / "splat.ply",
        nerf_dir.parent.parent.parent / f"{scene_id}.ply",  # <synset>/<scene_id>.ply
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _parse_ply_header(f) -> Tuple[Dict[str, object], int]:
    """Return (header_info, header_byte_count). Assumes file opened in binary mode at start."""
    magic = f.readline()
    if magic.strip() != b"ply":
        raise ValueError("Not a PLY file")

    fmt = None
    vertex_count = None
    props: List[Tuple[str, str]] = []  # (type, name)
    in_vertex = False
    header_lines = [magic]

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF in PLY header")
        header_lines.append(line)
        text = line.decode("ascii", errors="replace").strip()
        if text.startswith("format "):
            fmt = text.split()[1]
        elif text.startswith("element "):
            parts = text.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif text.startswith("property ") and in_vertex:
            parts = text.split()
            # property <type> <name>  (skip list properties)
            if parts[1] == "list":
                raise ValueError("List properties are not supported")
            props.append((parts[1], parts[2]))
        elif text == "end_header":
            break

    if fmt is None or vertex_count is None:
        raise ValueError("Invalid PLY header")
    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "properties": props,
    }, sum(len(x) for x in header_lines)


_TYPE_TO_STRUCT = {
    "char": "b",
    "uchar": "B",
    "int8": "b",
    "uint8": "B",
    "short": "h",
    "ushort": "H",
    "int16": "h",
    "uint16": "H",
    "int": "i",
    "uint": "I",
    "int32": "i",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def read_gaussian_ply_xyz_rgb(ply_path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Read xyz (+ optional SH DC / RGB) from a 3DGS / Splatfacto PLY."""
    with open(ply_path, "rb") as f:
        info, _ = _parse_ply_header(f)
        if info["format"] != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {info['format']} ({ply_path})")

        props: List[Tuple[str, str]] = info["properties"]  # type: ignore
        names = [n for _, n in props]
        fmt = "<" + "".join(_TYPE_TO_STRUCT[t] for t, _ in props)
        stride = struct.calcsize(fmt)
        count = int(info["vertex_count"])

        idx = {name: i for i, name in enumerate(names)}
        for required in ("x", "y", "z"):
            if required not in idx:
                raise ValueError(f"PLY missing '{required}': {ply_path}")

        has_sh = all(k in idx for k in ("f_dc_0", "f_dc_1", "f_dc_2"))
        has_rgb = all(k in idx for k in ("red", "green", "blue"))

        xyz: List[Tuple[float, float, float]] = []
        rgb: List[Tuple[int, int, int]] = []
        data = f.read(stride * count)
        if len(data) < stride * count:
            raise ValueError(f"Truncated PLY body: {ply_path}")

        for i in range(count):
            vals = struct.unpack_from(fmt, data, i * stride)
            x, y, z = float(vals[idx["x"]]), float(vals[idx["y"]]), float(vals[idx["z"]])
            xyz.append((x, y, z))
            if has_rgb:
                r = int(vals[idx["red"]])
                g = int(vals[idx["green"]])
                b = int(vals[idx["blue"]])
            elif has_sh:
                r = int(max(0.0, min(1.0, float(vals[idx["f_dc_0"]]) * SH_C0 + 0.5)) * 255)
                g = int(max(0.0, min(1.0, float(vals[idx["f_dc_1"]]) * SH_C0 + 0.5)) * 255)
                b = int(max(0.0, min(1.0, float(vals[idx["f_dc_2"]]) * SH_C0 + 0.5)) * 255)
            else:
                r = g = b = 128
            rgb.append((r, g, b))

    return xyz, rgb


def write_points3d_ply(out_path: Path, xyz: List[Tuple[float, float, float]], rgb: List[Tuple[int, int, int]]) -> None:
    """Write a simple point cloud PLY that SRGS `fetchPly` can load."""
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(out_path, "wb") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(struct.pack("<ffffffBBB", x, y, z, 0.0, 0.0, 0.0, r, g, b))


def copy_or_link(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if link:
        os.symlink(src.resolve(), dst)
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def convert_scene(scene_id: str, nerf_dir: Path, out_root: Path, link: bool) -> Path:
    out_dir = out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("transforms_train.json", "transforms_test.json"):
        src = nerf_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_dir / name)

    # Optional extras for debugging / alternate splits
    for name in ("transforms_val.json", "dataset_meta.json", "view_split.json"):
        src = nerf_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    train_src = nerf_dir / "train"
    if not train_src.is_dir():
        raise FileNotFoundError(train_src)
    copy_or_link(train_src, out_dir / "train", link=link)

    splat = find_splat_ply(nerf_dir, scene_id)
    points_path = out_dir / "points3d.ply"
    if splat is not None:
        xyz, rgb = read_gaussian_ply_xyz_rgb(splat)
        write_points3d_ply(points_path, xyz, rgb)
        init_src = str(splat)
    else:
        # Fallback: SRGS will synthesize random points if this is missing; create an empty marker note instead.
        init_src = "NONE (SRGS will random-init)"
        if points_path.exists():
            points_path.unlink()

    meta = {
        "scene_id": scene_id,
        "source_nerf_dataset": str(nerf_dir.resolve()),
        "init_ply_source": init_src,
        "white_background": True,
        "recommended_train_args": {
            "resolution": 4,
            "white_background": True,
            "eval": True,
            "note": (
                "Images are 400x400. Use -r 4 so SRGS trains on 100x100 LR, "
                "upscales x4 with SwinIR to 400, and evaluates against the native 400 GT."
            ),
        },
    }
    with open(out_dir / "srgs_scene_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SplatFormer LR nerf_datasets to SRGS Blender format")
    parser.add_argument("--src", type=str, default="lr_data_splatformer", help="Root of lr_data_splatformer")
    parser.add_argument("--out", type=str, default="data/splatformer", help="Output directory")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy train/ images instead of symlinking (default: symlink)",
    )
    parser.add_argument("--scene", type=str, default=None, help="Optional single scene id to convert")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src.resolve()}")

    scenes = discover_scenes(src)
    if args.scene:
        scenes = [(sid, path) for sid, path in scenes if sid == args.scene]
        if not scenes:
            raise SystemExit(f"Scene not found: {args.scene}")

    if not scenes:
        raise SystemExit(f"No nerf_dataset scenes found under {src}")

    out.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(scenes)} scene(s). Writing to {out.resolve()}")

    converted = []
    for scene_id, nerf_dir in scenes:
        out_dir = convert_scene(scene_id, nerf_dir, out, link=not args.copy_images)
        n_imgs = len(list((out_dir / "train").glob("*.png")))
        has_ply = (out_dir / "points3d.ply").is_file()
        print(f"  {scene_id}: images={n_imgs}, points3d.ply={'yes' if has_ply else 'no'} -> {out_dir}")
        converted.append(scene_id)

    with open(out / "scenes.txt", "w") as f:
        for sid in converted:
            f.write(sid + "\n")

    print("Done.")
    print("Train example:")
    print(
        f"  python train.py -s {out / converted[0]} -m output/{converted[0]} "
        f"-r 4 --white_background --eval"
    )


if __name__ == "__main__":
    main()
