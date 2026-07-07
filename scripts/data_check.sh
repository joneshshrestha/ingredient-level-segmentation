#!/usr/bin/env bash
# Run the FoodSeg103 data sanity check (split sizes, missing/orphan files,
# derived num_labels, unique mask values, class distribution).
#   Usage:  bash scripts/data_check.sh [--full-scan] [--dataset-root <path>]
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

python -m src.data --config configs/segformer_b0.yaml "$@"
