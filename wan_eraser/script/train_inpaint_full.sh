#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/conda/etc/profile.d/conda.sh
conda activate eidt

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL 调试和超时优化
export NCCL_DEBUG=WARN  # 改为WARN减少日志噪音，出问题时改回INFO
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600  # 1小时超时
export TORCH_DISTRIBUTED_DEBUG=OFF  # 关闭详细调试减少开销

cd "${ROOT_DIR}"

# 创建日志目录
LOG_DIR="${ROOT_DIR}/outputs/sft_1230/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo "Training log will be saved to: ${LOG_FILE}"
echo "To monitor: tail -f ${LOG_FILE}"

# Use DeepSpeed ZeRO-2 with CPU offload for full parameter training
# 使用 nohup 和重定向确保终端断开后继续运行
accelerate launch \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/deepspeed_zero2.json \
    train_wan_sft_inpaint_full.py --config configs/train_inpaint_full.yaml \
    2>&1 | tee "${LOG_FILE}"

echo "Training finished or interrupted. Check ${LOG_FILE} for details."
