#!/bin/bash

python train.py --dataset nuscenes \
    --model-dir models/nusc/1s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 2 \
    --n-output 2 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 4 --num-workers 4 \
    --num-epoch 15

python train.py --dataset nuscenes \
    --model-dir models/nusc/3s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 6 \
    --n-output 6 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 8 --num-workers 8 \
    --num-epoch 15

python train.py --dataset kitti \
    --model-dir models/kitti/1s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 5 --input-step 2 \
    --n-output 5 --output-step 2 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 8 --num-workers 8 \
    --num-epoch 15

python train.py --dataset kitti \
    --model-dir models/kitti/3s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 5 --input-step 6 \
    --n-output 5 --output-step 6 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 8 --num-workers 8 \
    --num-epoch 15

python train.py --dataset argoverse2 \
    --model-dir models/av2/1s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 5 --input-step 2 \
    --n-output 5 --output-step 2 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 8 --num-workers 8 \
    --num-epoch 15

python train.py --dataset argoverse2 \
    --model-dir models/av2/3s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 5 --input-step 6 \
    --n-output 5 --output-step 6 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 8 --num-workers 8 \
    --num-epoch 15

# Marine dataset training examples
# Use the train_marine.sh wrapper script which sets up CUDA environment properly
# Example: nohup bash scripts/train_marine.sh [args] > training.log 2>&1 &

# 1-second forecasting
# nohup bash scripts/train_marine.sh --dataset marine \
#     --marine-root /home/pratyush/ISyE_Research/datasets/extracted/marine_philos \
#     --marine-cfg configs/marine.yaml \
#     --model-dir models/marine/1s_forecasting \
#     --model-type dynamic \
#     --model-name occ \
#     --loss-type l1 \
#     --n-input 5 --input-step 2 \
#     --n-output 5 --output-step 2 \
#     --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
#     --voxel-size 0.2 \
#     --batch-size 1 --num-workers 1 \
#     --num-epoch 15 \
#     --lr-start 5e-4 > training.log 2>&1 &

# 3-second forecasting
# nohup bash scripts/train_marine.sh --dataset marine \
#     --marine-root /home/pratyush/ISyE_Research/datasets/extracted/marine_philos \
#     --marine-cfg configs/marine.yaml \
#     --model-dir models/marine/3s_forecasting \
#     --model-type dynamic \
#     --model-name occ \
#     --loss-type l1 \
#     --n-input 5 --input-step 6 \
#     --n-output 5 --output-step 6 \
#     --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
#     --voxel-size 0.2 \
#     --batch-size 1 --num-workers 1 \
#     --num-epoch 15 \
#     --lr-start 5e-4 > training_3s.log 2>&1 &

