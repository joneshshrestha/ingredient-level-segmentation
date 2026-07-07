#!/usr/bin/env bash
# Full SegFormer-B0 fine-tuning on FoodSeg103.
#   Usage:  bash scripts/train_segformer.sh [--epochs N] [--batch-size N] [--resume <output_dir>]
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

# Pin to a single visible GPU if you like (the box has 2x A100, often shared):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_segformer.sh
python -m src.train --config configs/segformer_b0.yaml "$@"
