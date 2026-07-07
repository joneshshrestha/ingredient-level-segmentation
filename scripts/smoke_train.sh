#!/usr/bin/env bash
# Tiny end-to-end training run to verify the whole pipeline (downloads the
# pretrained backbone, trains on a few images for 1 epoch, validates, saves).
#   Usage:  bash scripts/smoke_train.sh [extra train.py args]
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

python -m src.train --config configs/segformer_b0.yaml --smoke-test "$@"
