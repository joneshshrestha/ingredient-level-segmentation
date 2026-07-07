#!/usr/bin/env bash
# Evaluate a trained SegFormer-B1 checkpoint on the official TEST split (or val).
#   Usage:  bash scripts/evaluate_segformer_b1.sh
#           CHECKPOINT=/path/to/best SPLIT=test bash scripts/evaluate_segformer_b1.sh
#           bash scripts/evaluate_segformer_b1.sh --num-vis 8     # also save figures
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg
# conda activation scripts may reference unset vars; relax nounset around them.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-foodseg}"
conda activate "$CONDA_ENV"
set -u

DEFAULT_CKPT="$(python -c "import yaml;print(yaml.safe_load(open('configs/segformer_b1.yaml'))['output_dir'])")/best"
CKPT="${CHECKPOINT:-$DEFAULT_CKPT}"
SPLIT="${SPLIT:-test}"
echo "Evaluating B1 checkpoint: $CKPT  (split=$SPLIT)"

python -m src.evaluate --config configs/segformer_b1.yaml \
    --checkpoint "$CKPT" --split "$SPLIT" "$@"
