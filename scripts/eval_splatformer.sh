#!/usr/bin/env bash
# Train + render + metrics for converted SplatFormer scenes.
# Also renders the precomputed LR splat.ply (no extra training) as baseline_lr.
#
# Prerequisites:
#   1) conda env from environment.yml (+ CUDA rasterizer submodules installed)
#   2) python scripts/convert_splatformer_lr.py --src lr_data_splatformer --out data/splatformer
#
# Usage:
#   bash scripts/eval_splatformer.sh
#   bash scripts/eval_splatformer.sh data/splatformer output/splatformer
#   SCENE_ID=02691156-1345cd9d0da6d149c6f6da58b133bae0 bash scripts/eval_splatformer.sh
#   SKIP_TRAIN=1 SCENE_ID=... bash scripts/eval_splatformer.sh data/splatformer output/splatformer_xyz

set -euo pipefail

DATA_ROOT="${1:-data/splatformer}"
OUT_ROOT="${2:-output/splatformer}"
ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:-4}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

resolve_lr_ply() {
  python - "$1" <<'PY'
from pathlib import Path
import json, sys
scene = Path(sys.argv[1]).resolve()
scene_id = scene.name
candidates = []
meta_path = scene / "srgs_scene_meta.json"
if meta_path.is_file():
    meta = json.loads(meta_path.read_text())
    src = meta.get("init_ply_source")
    if src and not str(src).startswith("NONE"):
        candidates.append(Path(src))
    nerf = meta.get("source_nerf_dataset")
    if nerf:
        nerf = Path(nerf)
        candidates.append(nerf.parent / "export" / "splat.ply")
        candidates.append(nerf.parent.parent.parent / f"{scene_id}.ply")
synset = scene_id.split("-")[0]
candidates.extend([
    Path("lr_data_splatformer") / synset / ".work" / scene_id / "export" / "splat.ply",
    Path("lr_data_splatformer") / synset / f"{scene_id}.ply",
])
for path in candidates:
    if path.is_file():
        print(path)
        sys.exit(0)
sys.exit(1)
PY
}

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT"
  echo "Run: python scripts/convert_splatformer_lr.py --src lr_data_splatformer --out $DATA_ROOT"
  exit 1
fi

scenes_file="$(mktemp)"
trap 'rm -f "$scenes_file"' EXIT

if [[ -n "${SCENE_ID:-}" ]]; then
  echo "$SCENE_ID" > "$scenes_file"
elif [[ -f "$DATA_ROOT/scenes.txt" ]]; then
  grep -v '^[[:space:]]*$' "$DATA_ROOT/scenes.txt" > "$scenes_file"
else
  for d in "$DATA_ROOT"/*; do
    [[ -d "$d" && -f "$d/transforms_train.json" ]] && basename "$d"
  done > "$scenes_file"
fi

if [[ ! -s "$scenes_file" ]]; then
  echo "No scenes found under $DATA_ROOT"
  exit 1
fi

mkdir -p "$OUT_ROOT"

while IFS= read -r scene; do
  [[ -z "$scene" ]] && continue
  src="$DATA_ROOT/$scene"
  model="$OUT_ROOT/$scene"
  echo "=============================="
  echo "Scene: $scene"
  echo "Source: $src"
  echo "Output: $model"
  echo "=============================="

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    python train.py \
      -s "$src" \
      -m "$model" \
      -r "$RESOLUTION" \
      --white_background \
      --eval \
      --iterations "$ITERATIONS" \
      --save_iterations "$ITERATIONS" \
      --test_iterations "$ITERATIONS"
  elif [[ ! -f "$model/cfg_args" ]]; then
    echo "SKIP_TRAIN=1 but missing $model/cfg_args — train this scene first."
    exit 1
  fi

  if [[ "$SKIP_TRAIN" != "1" ]] || [[ ! -d "$model/test" ]]; then
    python render.py -m "$model" -r "$RESOLUTION" --skip_train
  fi

  if lr_ply="$(resolve_lr_ply "$src")"; then
    echo "Baseline LR splat: $lr_ply"
    python render.py -m "$model" -r "$RESOLUTION" --skip_train --ply "$lr_ply" --method baseline_lr
  else
    echo "No precomputed LR splat.ply for $scene; skipping baseline_lr"
  fi

  python metrics.py -m "$model"
done < "$scenes_file"

echo "All done. Results: $OUT_ROOT/*/results.json"
