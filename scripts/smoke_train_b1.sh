#!/usr/bin/env bash
# Tiny end-to-end SegFormer-B1 training run to verify the pipeline fast.
#   Usage:  bash scripts/smoke_train_b1.sh [extra train.py args]
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

python -m src.train --config configs/segformer_b1.yaml --smoke-test "$@"
