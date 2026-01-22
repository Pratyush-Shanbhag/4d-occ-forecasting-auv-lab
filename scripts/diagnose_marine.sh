#!/bin/bash
# Diagnostic wrapper script for marine dataset with proper CUDA environment setup

# Set CUDA environment variables
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to project directory
cd "$(dirname "$0")/.."

# Run diagnostic script with conda environment
exec conda run -n ff-recovery python -u scripts/diagnose_predictions.py "$@"


