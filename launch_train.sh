#!/bin/bash
# Launch script for multi-GPU distributed training of FaceXFormer-main
# 
# Usage:
#   Single GPU:   bash launch_train.sh
#   Multi-GPU:    bash launch_train.sh --nproc_per_node=4
#   
# Environment variables:
#   CUDA_VISIBLE_DEVICES: Specify which GPUs to use (e.g., "0,1,2,3")

set -e

# Default to all available GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "======================================"
    echo "Single GPU Training"
    echo "======================================"
    python train_simple.py
else
    echo "======================================"
    echo "Multi-GPU Distributed Training"
    echo "Number of GPUs: $NUM_GPUS"
    echo "======================================"
    
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        train.py \
        "$@"
fi
