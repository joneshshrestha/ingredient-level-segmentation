#!/usr/bin/env bash
# Full SegFormer-B1 fine-tuning on FoodSeg103.
# Writes to outputs/segformer_b1/ (configured in configs/segformer_b1.yaml);
# the completed B0 run in outputs/segformer_b0/ is never touched.
#   Usage:  bash scripts/train_segformer_b1.sh [--epochs N] [--batch-size N] [--resume <output_dir>]
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

# Pick a GPU externally, e.g.:  CUDA_VISIBLE_DEVICES=1 bash scripts/train_segformer_b1.sh
python -m src.train --config configs/segformer_b1.yaml "$@"
