#!/usr/bin/env bash
# Predict ingredient masks + overlays + visible-area % with a SegFormer-B1 checkpoint.
# Defaults the output to results/predictions_20_b1/ so it does not clash with the B0 set.
#   Usage:  INPUT=/path/to/image_or_folder bash scripts/predict_examples_b1.sh
#           CHECKPOINT=/path/to/best INPUT=imgs/ OUTPUT_DIR=results/predictions_20_b1 bash scripts/predict_examples_b1.sh
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
OUTPUT_DIR="${OUTPUT_DIR:-results/predictions_20_b1}"
: "${INPUT:?Set INPUT=<image file or folder of images>}"
echo "Predicting (B1) with checkpoint: $CKPT  on: $INPUT  ->  $OUTPUT_DIR"

python -m src.predict --config configs/segformer_b1.yaml \
    --checkpoint "$CKPT" --input "$INPUT" --output-dir "$OUTPUT_DIR" "$@"
