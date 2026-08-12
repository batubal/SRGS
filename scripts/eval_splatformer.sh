#!/usr/bin/env bash
# Train + render + metrics for converted SplatFormer scenes.
#
# Prerequisites:
#   1) conda env from environment.yml (+ CUDA rasterizer submodules installed)
#   2) python scripts/convert_splatformer_lr.py --src lr_data_splatformer --out data/splatformer
#
# Usage:
#   bash scripts/eval_splatformer.sh
#   bash scripts/eval_splatformer.sh data/splatformer output/splatformer
#   SCENE_ID=02691156-1345cd9d0da6d149c6f6da58b133bae0 bash scripts/eval_splatformer.sh

set -euo pipefail

DATA_ROOT="${1:-data/splatformer}"
OUT_ROOT="${2:-output/splatformer}"
ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:-4}"

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

  python train.py \
    -s "$src" \
    -m "$model" \
    -r "$RESOLUTION" \
    --white_background \
    --eval \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --test_iterations "$ITERATIONS"

  python render.py -m "$model" -r "$RESOLUTION" --skip_train
  python metrics.py -m "$model"
done < "$scenes_file"

echo "All done. Results: $OUT_ROOT/*/results.json"
