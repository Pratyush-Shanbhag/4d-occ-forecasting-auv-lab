#!/bin/bash
# Testing wrapper script for marine dataset with proper CUDA environment setup
# Automatically enables point cloud output generation

# Set CUDA environment variables
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to project directory
cd "$(dirname "$0")/.."

# Check if --write-dense-pointcloud is already in arguments
if [[ "$*" == *"--write-dense-pointcloud"* ]]; then
    # Flag already present, use as-is
    exec conda run -n ff-recovery python -u test.py "$@"
else
    # Automatically add --write-dense-pointcloud flag to save point clouds
    exec conda run -n ff-recovery python -u test.py "$@" --write-dense-pointcloud
fi

