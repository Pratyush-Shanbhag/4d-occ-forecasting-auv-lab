#!/bin/bash
SPLIT="test"
BATCH_SIZE=2
NUM_WORKERS=2

python test.py --model-dir models/kitti/1s_forecasting \
    --test-epoch 14 \
    --test-split $SPLIT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --compute-chamfer-dist

# Marine dataset testing example
# Note: Update model config to include marine_root path, or set via environment
# The model config.json should have "marine_root" and "dataset": "marine" fields
# python test.py --model-dir models/marine/1s_forecasting \
#     --test-epoch 14 \
#     --test-split $SPLIT \
#     --batch-size $BATCH_SIZE \
#     --num-workers $NUM_WORKERS \
#     --compute-chamfer-dist
