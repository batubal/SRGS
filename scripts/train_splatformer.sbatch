#!/bin/bash
#SBATCH -J srgs
#SBATCH -p <gpu-partition>   # e.g. gpu / a100 / akcil — set to your cluster partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

module purge
module load miniconda3
module load lib/cuda/12.4
module load comp/gcc/12.3.0

eval "$(conda shell.bash hook)"

export SCRATCH=${SCRATCH:-/arf/scratch/$USER}
export CONDA_ENVS_PATH=$SCRATCH/conda/envs
export CONDA_PKGS_DIRS=$SCRATCH/conda/pkgs
export PIP_CACHE_DIR=$SCRATCH/pip-cache

export CUDA_HOME=${CUDA_HOME:?CUDA_HOME not set by cuda module}
export PATH="$CUDA_HOME/bin:$PATH"

conda activate "$SCRATCH/conda/envs/srgs"

# Required for simple_knn / diff-gaussian-rasterization .so (libc10.so)
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export SRGS_ROOT="${SRGS_ROOT:-$HOME/repos/SRGS}"
cd "$SRGS_ROOT"

# Sanity check (fail early if env is broken)
python - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
assert torch.cuda.is_available(), "CUDA not available in this job"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("extensions OK")
PY

# Convert once on login/CPU node if needed:
# python scripts/convert_splatformer_lr.py --src lr_data_splatformer --out data/splatformer

SCENE="${SCENE:-02691156-1345cd9d0da6d149c6f6da58b133bae0}"
DATA_ROOT="${DATA_ROOT:-data/splatformer}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/srgs_out}"
RESOLUTION="${RESOLUTION:-4}"
ITERATIONS="${ITERATIONS:-30000}"

mkdir -p "$OUT_ROOT/$SCENE"

echo "Scene: $SCENE"
echo "Source: $DATA_ROOT/$SCENE"
echo "Output: $OUT_ROOT/$SCENE"

python train.py \
  -s "$DATA_ROOT/$SCENE" \
  -m "$OUT_ROOT/$SCENE" \
  -r "$RESOLUTION" \
  --white_background \
  --eval \
  --iterations "$ITERATIONS" \
  --save_iterations "$ITERATIONS" \
  --test_iterations "$ITERATIONS"

python render.py -m "$OUT_ROOT/$SCENE" -r "$RESOLUTION" --skip_train
python metrics.py -m "$OUT_ROOT/$SCENE"

echo "Done. Results: $OUT_ROOT/$SCENE/results.json"
