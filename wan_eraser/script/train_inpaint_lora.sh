#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

cd "${ROOT_DIR}"


accelerate launch --num_processes 8 train_wan_sft_inpaint.py --config configs/train_inpaint_lora.yaml
