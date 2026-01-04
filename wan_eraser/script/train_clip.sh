#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set GPU devices (modify as needed)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Number of processes (should match GPU count)
NUM_PROCESSES=8

cd "${ROOT_DIR}"

accelerate launch \
    --num_processes ${NUM_PROCESSES} \
    --mixed_precision bf16 \
    train_wan_sft_clip.py \
    --config configs/train_clip.yaml

echo "Training completed!"
