# Training Inputs and Testing Outputs Guide

This document explains the training inputs and testing outputs for the Point Cloud Forecasting model adapted for the marine dataset.

## Table of Contents
1. [Training Inputs](#training-inputs)
2. [Training Process](#training-process)
3. [Testing Outputs](#testing-outputs)
4. [Output File Structure](#output-file-structure)
5. [Interpreting Results](#interpreting-results)

---

## Training Inputs

### Data Format

The model is trained on the **Philos – Stationary, Passed By Launch** marine dataset, which contains LiDAR point cloud data from a stationary vessel observing ships passing by.

### Input Data Structure

#### 1. **Point Cloud Frames**
- **Format**: Binary files (`.bin`) in KITTI-like format
- **Location**: `{marine_root}/sequences/00/velodyne/`
- **Structure**: Each file contains point cloud data as a flat array of `float32` values
- **Shape**: `(N, 4)` where N is the number of points
  - Columns: `[x, y, z, intensity]`
  - Coordinates are in the LiDAR sensor frame

#### 2. **Pose Data**
- **Format**: Text file (`poses.txt`)
- **Location**: `{marine_root}/sequences/00/poses.txt`
- **Structure**: One 4×4 transformation matrix per line (12 values, flattened)
- **Purpose**: Global pose of the sensor platform (R/V Philos) for each frame
- **Format**: `[r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]` (rotation matrix + translation)

#### 3. **Calibration Data**
- **Format**: Text file (`calib.txt`)
- **Location**: `{marine_root}/sequences/00/calib.txt`
- **Purpose**: Transformation from LiDAR to ego vehicle frame (identity for marine dataset)

### Training Input Parameters

The model receives the following inputs during training:

#### Temporal Inputs
- **`n_input`**: Number of past frames (default: 5)
- **`input_step`**: Step size between input frames (default: 2)
  - Example: With `input_step=2`, frames are sampled every 2 steps
- **`n_output`**: Number of future frames to predict (default: 5)
- **`output_step`**: Step size between output frames (default: 2)

#### Spatial Parameters
- **`pc_range`**: Point cloud range in meters
  - Format: `[x_min, y_min, z_min, x_max, y_max, z_max]`
  - Default: `[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]`
  - Points outside this range are filtered out
- **`voxel_size`**: Voxel resolution in meters (default: 0.2)
  - Determines the discretization of 3D space
  - Grid size = `(range_size / voxel_size)³`

#### Data Preprocessing

1. **Ego Vehicle Filtering**: Points from the R/V Philos vessel are removed
   - Vessel dimensions: ~7.6m (length) × ~2.7m (width)
   - Filtered in KITTI coordinate system before transformation

2. **Coordinate Transformation**:
   - Points are transformed from LiDAR frame to a reference frame
   - Then converted to nuScenes coordinate system (used internally by the model)
   - Transformation: KITTI → nuScenes (handled by `KittiPoints2nuScenes`)

3. **Range Filtering**: Points outside `pc_range` are removed after coordinate transformation

### Training Data Flow

```
Raw LiDAR Frame (.bin)
    ↓
Load point cloud (x, y, z, intensity)
    ↓
Filter ego vehicle points
    ↓
Transform to reference frame (using poses)
    ↓
Convert to nuScenes coordinate system
    ↓
Filter by pc_range
    ↓
Model Input: [input_points, input_tindex, output_origin, output_points, output_tindex]
```

### Model Input Tensors

For each training sample, the model receives:

1. **`input_points`**: Tensor of shape `(N_in, 3)`
   - Past point clouds concatenated
   - N_in = total points across all input frames

2. **`input_tindex`**: Tensor of shape `(N_in,)`
   - Time indices for each point (0 to n_input-1)
   - Indicates which input frame each point belongs to

3. **`output_origin`**: Tensor of shape `(n_output, 3)`
   - Origin positions for each future frame
   - Used for ray casting during rendering

4. **`output_points`**: Tensor of shape `(N_out, 3)`
   - Ground truth future point clouds (for loss computation)
   - N_out = total points across all output frames

5. **`output_tindex`**: Tensor of shape `(N_out,)`
   - Time indices for each output point (0 to n_output-1)

---

## Training Process

### Training Configuration

- **Model Type**: `dynamic` (temporal occupancy forecasting)
- **Loss Type**: `l1` (L1 loss for distance prediction)
- **Batch Size**: 1 (optimized for GPU memory)
- **Learning Rate**: 5e-4 (initial)
- **Epochs**: 15
- **Optimizer**: Adam (default)

### Training Output

During training, the model:
1. Predicts future occupancy volumes
2. Renders predicted point clouds via differentiable voxel rendering (DVR)
3. Computes L1 loss between predicted and ground truth ray distances
4. Updates model parameters via backpropagation

### Checkpoints

Model checkpoints are saved in:
- **Location**: `models/marine/1s_forecasting/ckpts/`
- **Format**: `model_epoch_{epoch}.pth` (final checkpoint per epoch)
- **Intermediate**: `model_epoch_{epoch}_iter_{iteration}.pth` (periodic saves)

---

## Testing Outputs

### Test Process

During testing, the model:
1. Loads a trained checkpoint (e.g., `model_epoch_14.pth`)
2. Processes test split data (151 frames for marine dataset)
3. Generates predictions for future frames
4. Computes evaluation metrics
5. Saves point clouds for visualization

### Output Files

#### 1. Metrics Log File

**Location**: `models/marine/1s_forecasting/results/test/epoch_{epoch}/{timestamp}.txt`

**Contents**:
```
Final Chamfer Distance: tensor(47.1815, device='cuda:0')
Final Chamfer Distance Inner: tensor(43.4032, device='cuda:0')
Final L1 Error: 5.054173952924218
Final AbsRel Error: 1.5929600763497056
```

**Metrics Explained**:
- **Chamfer Distance**: Average bidirectional distance between predicted and ground truth point clouds (lower is better)
- **Chamfer Distance Inner**: Variant focusing on inner regions of point clouds
- **L1 Error**: Mean absolute error in predicted ray distances (meters, lower is better)
- **AbsRel Error**: Mean absolute relative error `|pred - gt| / gt` (scale-invariant, lower is better)

#### 2. Point Cloud Files

**Location**: `models/marine/1s_forecasting/results/test/epoch_{epoch}/pointclouds/`

**Format**: PLY (Polygon File Format) - standard 3D point cloud format

**File Naming Convention**:
- Predicted: `batch_{batch_idx:04d}_sample_{sample_idx:04d}_time_{time_idx:02d}_pred.ply`
- Ground Truth: `batch_{batch_idx:04d}_sample_{sample_idx:04d}_time_{time_idx:02d}_gt.ply`

**Example Files**:
- `batch_0000_sample_0000_time_00_pred.ply` - Predicted point cloud for batch 0, sample 0, time step 0
- `batch_0000_sample_0000_time_00_gt.ply` - Ground truth point cloud for the same

**File Structure**:
- Each PLY file contains 3D points with (x, y, z) coordinates
- Can be opened in:
  - CloudCompare
  - MeshLab
  - Open3D (Python library)
  - Blender
  - Any PLY-compatible viewer

#### 3. Test Log File

**Location**: `test_output.log` (in project root)

**Contents**: Full console output from testing, including:
- Per-batch progress
- Intermediate metrics
- Final aggregated metrics
- Any warnings or errors

---

## Output File Structure

```
models/marine/1s_forecasting/
├── config.json                    # Training configuration
├── ckpts/                         # Model checkpoints
│   ├── model_epoch_0.pth
│   ├── model_epoch_1.pth
│   └── ...
└── results/
    └── test/
        └── epoch_14/
            ├── {timestamp}.txt   # Metrics log
            └── pointclouds/      # Point cloud outputs
                ├── batch_0000_sample_0000_time_00_pred.ply
                ├── batch_0000_sample_0000_time_00_gt.ply
                ├── batch_0000_sample_0000_time_01_pred.ply
                ├── batch_0000_sample_0000_time_01_gt.ply
                └── ...
```

---

## Interpreting Results

### Metrics Interpretation

#### Chamfer Distance
- **What it measures**: Overall shape similarity between predicted and ground truth point clouds
- **Good values**: Lower is better (typically < 50 for this dataset)
- **Current result**: 47.18 (moderate performance)

#### L1 Error
- **What it measures**: Average error in predicted ray distances (in meters)
- **Good values**: Lower is better (typically < 10 meters)
- **Current result**: 5.05 meters (reasonable for marine environment)

#### Absolute Relative Error
- **What it measures**: Scale-invariant relative error
- **Good values**: Lower is better (typically < 2.0)
- **Current result**: 1.59 (acceptable performance)

### Visualizing Point Clouds

To compare predicted vs ground truth:

1. **Using CloudCompare**:
   ```bash
   cloudcompare batch_0000_sample_0000_time_00_pred.ply batch_0000_sample_0000_time_00_gt.ply
   ```

2. **Using Python (Open3D)**:
   ```python
   import open3d as o3d
   
   pred = o3d.io.read_point_cloud("batch_0000_sample_0000_time_00_pred.ply")
   gt = o3d.io.read_point_cloud("batch_0000_sample_0000_time_00_gt.ply")
   
   pred.paint_uniform_color([1, 0, 0])  # Red for predicted
   gt.paint_uniform_color([0, 1, 0])    # Green for ground truth
   
   o3d.visualization.draw_geometries([pred, gt])
   ```

### What to Look For

When visualizing point clouds, check:
1. **Shape similarity**: Do predicted clouds match the overall shape of ground truth?
2. **Density**: Are predicted clouds too sparse or too dense?
3. **Temporal consistency**: Do predictions evolve smoothly across time steps?
4. **Missing objects**: Are there objects in ground truth that are missing in predictions?
5. **False positives**: Are there predicted points where there shouldn't be any?

---

## Dataset Statistics

### Marine Dataset (Philos)

- **Total frames**: ~1000 frames
- **Train split**: ~70% (700 frames)
- **Val split**: ~15% (150 frames)
- **Test split**: ~15% (151 frames)
- **Temporal resolution**: Variable (depends on LiDAR frame rate)
- **Point cloud density**: Variable (typically 1000-10000 points per frame)

### Coordinate Systems

1. **LiDAR Frame**: Original sensor coordinates
2. **KITTI Frame**: Standardized format (forward=X, left=Y, up=Z)
3. **nuScenes Frame**: Model's internal coordinate system (forward=X, right=Y, up=Z)

---

## Running Training and Testing

### Training Command

```bash
nohup bash scripts/train_marine.sh \
    --dataset marine \
    --marine-root /path/to/extracted/marine/data \
    --marine-cfg configs/marine.yaml \
    --model-dir models/marine/1s_forecasting \
    --model-type dynamic \
    --model-name occ \
    --loss-type l1 \
    --n-input 5 --input-step 2 \
    --n-output 5 --output-step 2 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 1 \
    --num-workers 1 \
    --num-epoch 15 \
    --lr-start 5e-4 \
    > training.log 2>&1 &
```

### Testing Command

```bash
nohup bash scripts/test_marine.sh \
    --model-dir models/marine/1s_forecasting \
    --test-epoch 14 \
    --test-split test \
    --batch-size 2 \
    --num-workers 2 \
    --compute-chamfer-distance \
    > test_output.log 2>&1 &
```

**Note**: The `test_marine.sh` script automatically enables `--write-dense-pointcloud` to save point cloud outputs.

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or `pc_range`
2. **Empty Point Clouds**: Check if `pc_range` is too restrictive
3. **Poor Metrics**: Verify data preprocessing and coordinate transformations
4. **Missing Point Clouds**: Ensure `--write-dense-pointcloud` flag is set (automatic in `test_marine.sh`)

---

## References

- Original paper: "Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting"
- Dataset: MIT Sea Grant Philos dataset
- Model architecture: OccupancyForecastingNetwork with differentiable voxel rendering

---

*Last updated: January 2026*

