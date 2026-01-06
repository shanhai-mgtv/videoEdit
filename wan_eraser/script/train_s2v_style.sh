#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configuration
CONFIG_FILE="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/configs/train_inpaint_s2v_style.yaml"
NUM_GPUS=8

# Run training with accelerate
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    --multi_gpu \
    /mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/train_wan_sft_s2v_style.py \
    --config_path ${CONFIG_FILE}
