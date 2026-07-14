#!/bin/bash
# =============================================================================
# NeFIC Inference Script
#
# Usage:
#   bash run_test.sh <checkpoint_dir> [dataset] [model_path] [--entropy]
#
# Examples:
#   bash run_test.sh /path/to/checkpoint
#   bash run_test.sh /path/to/checkpoint kodak
#   bash run_test.sh /path/to/checkpoint clic /path/to/CogVideoX1.5-5B-standard
#   bash run_test.sh /path/to/checkpoint kodak "" --entropy
# =============================================================================

set -e
export CUDA_VISIBLE_DEVICES=0

# Parse --entropy flag from any position
USE_ENTROPY=""
POSITIONAL=()
for arg in "$@"; do
    if [ "$arg" = "--entropy" ]; then
        USE_ENTROPY="--use_entropy_coding"
    else
        POSITIONAL+=("$arg")
    fi
done

# Arguments
CHECKPOINT_DIR=${POSITIONAL[0]:?"Usage: bash run_test.sh <checkpoint_dir> [dataset] [model_path] [--entropy]"}
DATASET=${POSITIONAL[1]:-"kodak"}
MODEL_PATH=${POSITIONAL[2]:-"/path/to/CogVideoX1.5-5B-standard"}

# Dataset paths (modify as needed)
if [ "$DATASET" = "kodak" ]; then
    INPUT_DIR="/path/to/kodak"
elif [ "$DATASET" = "clic" ]; then
    INPUT_DIR="/path/to/clic"
elif [ "$DATASET" = "div2k" ]; then
    INPUT_DIR="/path/to/DIV2K_valid_HR"
else
    INPUT_DIR="$DATASET"  # Allow custom path
fi

OUTPUT_DIR="./results_${DATASET}_$(basename $(dirname $CHECKPOINT_DIR))_$(basename $CHECKPOINT_DIR)"

echo "============================================"
echo " NeFIC Inference"
echo " Dataset:    $DATASET"
echo " Model:      $MODEL_PATH"
echo " Checkpoint: $CHECKPOINT_DIR"
echo " Input:      $INPUT_DIR"
echo " Output:     $OUTPUT_DIR"
echo " Entropy:    ${USE_ENTROPY:-"off (STE simulation)"}"
echo "============================================"

cd "$(dirname "$0")"

python test.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dtype "bfloat16" \
    --pad_multiple 64 \
    --with_color_fix \
    --save_intermediate \
    $USE_ENTROPY

# ---- FID/KID evaluation for clic and div2k (requires >50 images) ----
if [ "$DATASET" = "clic" ] || [ "$DATASET" = "div2k" ]; then
    EVAL_SCRIPT="$(dirname "$0")/eval_fid_kid.py"

    echo ""
    echo "============================================"
    echo " FID/KID Evaluation: $DATASET"
    echo "============================================"
    python "$EVAL_SCRIPT" \
        --recon_dir "$OUTPUT_DIR" \
        --gt_dir "$INPUT_DIR" | tee "${OUTPUT_DIR}/fid_kid_results.txt"
fi
