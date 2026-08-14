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

  # Patch already-converted scenes (disjoint splits + xyz-only splat init):
  python scripts/convert_splatformer_lr.py --fix-existing --out data/splatformer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# fetchPly requires RGB; 128/255 = 0.5 maps to ~0 SH DC in create_from_pcd.
NEUTRAL_RGB = (128, 128, 128)


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


def read_gaussian_ply_xyz(ply_path: Path) -> List[Tuple[float, float, float]]:
    """Read only xyz from a 3DGS / Splatfacto PLY. Ignores SH, opacity, scale, rotation."""
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

        data = f.read(stride * count)
        if len(data) < stride * count:
            raise ValueError(f"Truncated PLY body: {ply_path}")

        xyz: List[Tuple[float, float, float]] = []
        for i in range(count):
            vals = struct.unpack_from(fmt, data, i * stride)
            xyz.append((float(vals[idx["x"]]), float(vals[idx["y"]]), float(vals[idx["z"]])))
        return xyz


def write_points3d_ply(out_path: Path, xyz: List[Tuple[float, float, float]]) -> None:
    """Write xyz-only init: positions from splat, neutral RGB, zero normals.

    SRGS create_from_pcd still resets scale / rotation / opacity / SH-rest;
    skipping splat RGB/SH avoids copying appearance that saw eval views.
    """
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
    r, g, b = NEUTRAL_RGB
    with open(out_path, "wb") as f:
        f.write(header)
        for x, y, z in xyz:
            f.write(struct.pack("<ffffffBBB", x, y, z, 0.0, 0.0, 0.0, r, g, b))


def write_xyz_only_init(splat: Path, points_path: Path) -> int:
    xyz = read_gaussian_ply_xyz(splat)
    write_points3d_ply(points_path, xyz)
    return len(xyz)


def resolve_splat_ply(scene_dir: Path) -> Optional[Path]:
    meta_path = scene_dir / "srgs_scene_meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    src = meta.get("init_ply_source")
    if src and not str(src).startswith("NONE"):
        path = Path(src)
        if path.is_file():
            return path
    nerf = meta.get("source_nerf_dataset")
    if nerf:
        found = find_splat_ply(Path(nerf), scene_dir.name)
        if found is not None:
            return found
    return None


def _frame_path_key(file_path: str) -> str:
    path = file_path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if path.endswith(".png"):
        path = path[:-4]
    return path


def _frame_pose_key(frame: dict) -> str:
    return json.dumps(frame.get("transform_matrix"), separators=(",", ":"))


def enforce_disjoint_splits(scene_dir: Path) -> Tuple[int, int, int]:
    """Drop any train frame that also appears in transforms_test.json.

    SplatFormer exports often set train_all_views=true, so eval cameras are a
    subset of train. SRGS --eval does not remove those overlapping train frames.

    Returns (n_train_kept, n_test, n_dropped).
    """
    train_path = scene_dir / "transforms_train.json"
    test_path = scene_dir / "transforms_test.json"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Missing transforms in {scene_dir}")

    train = json.loads(train_path.read_text())
    test = json.loads(test_path.read_text())
    test_frames = test.get("frames", [])
    train_frames = train.get("frames", [])

    test_paths = {_frame_path_key(f["file_path"]) for f in test_frames}
    test_poses = {_frame_pose_key(f) for f in test_frames}

    kept = []
    dropped = []
    for frame in train_frames:
        if _frame_path_key(frame["file_path"]) in test_paths or _frame_pose_key(frame) in test_poses:
            dropped.append(frame)
        else:
            kept.append(frame)

    if not kept:
        raise ValueError(f"All train frames overlap the test split in {scene_dir}")

    if dropped:
        all_path = scene_dir / "transforms_train_all.json"
        if not all_path.is_file():
            shutil.copy2(train_path, all_path)
        train["frames"] = kept
        with open(train_path, "w") as f:
            json.dump(train, f, indent=4)
            f.write("\n")

    dropped_paths = {_frame_path_key(f["file_path"]) for f in dropped}
    _rewrite_split_metadata(scene_dir, kept, test_frames, dropped_paths)

    srgs_meta_path = scene_dir / "srgs_scene_meta.json"
    if srgs_meta_path.is_file():
        srgs_meta = json.loads(srgs_meta_path.read_text())
        srgs_meta["disjoint_splits"] = True
        srgs_meta["train_frames"] = len(kept)
        srgs_meta["test_frames"] = len(test_frames)
        srgs_meta["dropped_overlapping_train_frames"] = len(dropped)
        with open(srgs_meta_path, "w") as f:
            json.dump(srgs_meta, f, indent=2)
            f.write("\n")

    return len(kept), len(test_frames), len(dropped)


def _rewrite_split_metadata(
    scene_dir: Path,
    train_frames: List[dict],
    test_frames: List[dict],
    dropped_paths: set,
) -> None:
    train_paths = [_frame_path_key(f["file_path"]) for f in train_frames]
    test_paths = [_frame_path_key(f["file_path"]) for f in test_frames]

    def _index_from_path(path: str) -> Optional[int]:
        name = Path(path).name
        if name.startswith("r_"):
            try:
                return int(name[2:])
            except ValueError:
                return None
        return None

    train_indices = []
    for path in train_paths:
        idx = _index_from_path(path)
        if idx is not None:
            train_indices.append(idx)
    eval_indices = []
    for path in test_paths:
        idx = _index_from_path(path)
        if idx is not None:
            eval_indices.append(idx)

    split_path = scene_dir / "view_split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text())
        split["train_indices"] = train_indices
        split["eval_indices"] = eval_indices
        split["disjoint"] = True
        split["dropped_from_train"] = sorted(dropped_paths)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
            f.write("\n")

    meta_path = scene_dir / "dataset_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["train_indices"] = train_indices
        meta["eval_indices"] = eval_indices
        meta["num_train_views"] = len(train_frames)
        meta["disjoint_splits"] = True
        meta["train_all_views"] = False
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")


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


def convert_scene(scene_id: str, nerf_dir: Path, out_root: Path, link: bool) -> Tuple[Path, Tuple[int, int, int]]:
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
    n_init_pts = 0
    if splat is not None:
        n_init_pts = write_xyz_only_init(splat, points_path)
        init_src = str(splat)
    else:
        # Fallback: SRGS will synthesize random points if this is missing; create an empty marker note instead.
        init_src = "NONE (SRGS will random-init)"
        if points_path.exists():
            points_path.unlink()

    n_train, n_test, n_dropped = enforce_disjoint_splits(out_dir)

    meta = {
        "scene_id": scene_id,
        "source_nerf_dataset": str(nerf_dir.resolve()),
        "init_ply_source": init_src,
        "init_attributes": "xyz_only",
        "init_points": n_init_pts,
        "white_background": True,
        "disjoint_splits": True,
        "train_frames": n_train,
        "test_frames": n_test,
        "dropped_overlapping_train_frames": n_dropped,
        "recommended_train_args": {
            "resolution": 4,
            "white_background": True,
            "eval": True,
            "note": (
                "Images are 400x400. Use -r 4 so SRGS trains on 100x100 LR, "
                "upscales x4 with SwinIR to 400, and evaluates against the native 400 GT. "
                "Overlapping eval views are stripped from transforms_train.json. "
                "points3d.ply keeps splat xyz only; SH/opacity/scale/rotation are not copied."
            ),
        },
    }
    with open(out_dir / "srgs_scene_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir, (n_train, n_test, n_dropped)


def discover_converted_scenes(out: Path) -> List[Path]:
    scenes = []
    for path in sorted(out.iterdir()):
        if path.is_dir() and (path / "transforms_train.json").is_file() and (path / "transforms_test.json").is_file():
            scenes.append(path)
    return scenes


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
    parser.add_argument(
        "--fix-existing",
        action="store_true",
        help="Strip overlapping train/test views in --out and rewrite points3d.ply as xyz-only",
    )
    args = parser.parse_args()

    out = Path(args.out)

    if args.fix_existing:
        if not out.is_dir():
            raise SystemExit(f"Converted data not found: {out.resolve()}")
        scenes = discover_converted_scenes(out)
        if args.scene:
            scenes = [p for p in scenes if p.name == args.scene]
            if not scenes:
                raise SystemExit(f"Scene not found in {out}: {args.scene}")
        if not scenes:
            raise SystemExit(f"No converted scenes found under {out}")
        print(f"Fixing splits and xyz-only init in {len(scenes)} scene(s) under {out.resolve()}")
        for scene_dir in scenes:
            n_train, n_test, n_dropped = enforce_disjoint_splits(scene_dir)
            splat = resolve_splat_ply(scene_dir)
            if splat is not None:
                n_pts = write_xyz_only_init(splat, scene_dir / "points3d.ply")
                init_note = f"xyz_only={n_pts} from {splat}"
            else:
                init_note = "points3d unchanged (no splat.ply)"
            meta_path = scene_dir / "srgs_scene_meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                meta["init_attributes"] = "xyz_only"
                if splat is not None:
                    meta["init_ply_source"] = str(splat)
                    meta["init_points"] = n_pts
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                    f.write("\n")
            print(
                f"  {scene_dir.name}: train={n_train} test={n_test} "
                f"dropped_overlap={n_dropped}; {init_note}"
            )
        print("Done. Retrain from scratch; do not reuse the leaked checkpoints.")
        return

    src = Path(args.src)
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
        out_dir, (n_train, n_test, n_dropped) = convert_scene(
            scene_id, nerf_dir, out, link=not args.copy_images
        )
        n_imgs = len(list((out_dir / "train").glob("*.png")))
        has_ply = (out_dir / "points3d.ply").is_file()
        print(
            f"  {scene_id}: images={n_imgs}, points3d.ply={'yes' if has_ply else 'no'}, "
            f"train={n_train} test={n_test} dropped_overlap={n_dropped} -> {out_dir}"
        )
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
