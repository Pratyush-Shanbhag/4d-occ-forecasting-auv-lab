#!/bin/bash
# Training wrapper script for marine dataset with proper CUDA environment setup

# Set CUDA environment variables
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Change to project directory
cd "$(dirname "$0")/.."

# Run training with conda environment (output will be captured by nohup)
# Use exec to ensure output is properly redirected
exec conda run -n ff-recovery python -u train.py "$@"

