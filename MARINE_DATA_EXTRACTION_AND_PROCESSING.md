# Marine LiDAR Data Extraction and Processing Guide

This document provides a detailed explanation of how the marine LiDAR data was extracted from the ROS bag file and processed to be compatible with the point cloud forecasting training pipeline.

## Table of Contents
1. [Overview](#overview)
2. [Data Source](#data-source)
3. [Extraction Process](#extraction-process)
4. [Data Format Conversion](#data-format-conversion)
5. [Processing Pipeline](#processing-pipeline)
6. [Coordinate System Transformations](#coordinate-system-transformations)
7. [Dataset Structure](#dataset-structure)
8. [Training Data Flow](#training-data-flow)

---

## Overview

The marine dataset (MIT Sea Grant Philos dataset) comes in ROS bag format, which contains:
- **LiDAR point clouds**: `sensor_msgs/PointCloud2` messages
- **Pose/odometry data**: `nav_msgs/Odometry` messages
- **Other sensor data**: IMU, radar, video (not used for this project)

The extraction and processing pipeline converts this ROS data into a **KITTI-like format** that is compatible with the existing codebase, which was originally designed for KITTI and nuScenes datasets.

---

## Data Source

### Input: ROS Bag File
- **File**: `section.bag` (from Philos – Stationary, Passed By Launch dataset)
- **Format**: ROS bag file (`.bag`)
- **Location**: `/home/pratyush/ISyE_Research/datasets/unzipped/auv_lab/philos_2020_11_06_stationary_passed_by_launch/section.bag`

### ROS Topics in Bag File
The extraction script automatically detects relevant topics:
- **LiDAR topic**: `/velodyne_points` (or auto-detected `PointCloud2` topic)
- **Pose topic**: `/lidar_odometry` (or auto-detected `Odometry`/`PoseStamped` topic)

---

## Extraction Process

### Script: `scripts/extract_marine_data.py`

The extraction process consists of several steps:

### Step 1: Topic Detection

The script automatically detects relevant ROS topics:

```python
# Auto-detect LiDAR topic
lidar_candidates = ['/velodyne_points', '/lidar/points', '/points', 
                   '/velodyne/points', '/ouster/points', '/livox/lidar']
# Falls back to any PointCloud2 topic if candidates not found

# Auto-detect pose topic
pose_candidates = ['/odom', '/imu/odom', '/pose', '/ego_pose', 
                  '/odometry/filtered', '/nav_msgs/Odometry']
# Falls back to any Odometry or PoseStamped topic
```

### Step 2: Message Extraction

#### LiDAR Point Cloud Extraction

```python
def extract_pointcloud(msg):
    """Extract point cloud from ROS PointCloud2 message."""
    points = []
    for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2], 1.0])  # x, y, z, intensity
    return np.array(points, dtype=np.float32)
```

**Process**:
1. Read `sensor_msgs/PointCloud2` messages
2. Extract `x`, `y`, `z` coordinates
3. Set intensity to `1.0` (default, as intensity may not be available)
4. Filter out NaN values
5. Store as `float32` numpy array

**Output format**: `(N, 4)` array where N is the number of points, columns are `[x, y, z, intensity]`

#### Pose/Odometry Extraction

```python
def pose_to_matrix(pose_msg):
    """Convert ROS pose message to 4x4 transformation matrix."""
    # Handle Odometry messages (pose.pose.pose)
    if hasattr(pose_msg, 'pose') and hasattr(pose_msg.pose, 'pose'):
        pose = pose_msg.pose.pose  # PoseWithCovariance
    # ... extract position (x, y, z) and orientation (quaternion qx, qy, qz, qw)
    # Convert quaternion to rotation matrix
    # Create 4x4 transformation matrix
    return T
```

**Process**:
1. Extract position: `(x, y, z)` from `pose.position`
2. Extract orientation: quaternion `(qx, qy, qz, qw)` from `pose.orientation`
3. Convert quaternion to 3×3 rotation matrix:
   ```
   R = [
       [1-2(qy²+qz²), 2(qxqy-qwqz), 2(qxqz+qwqy)],
       [2(qxqy+qwqz), 1-2(qx²+qz²), 2(qyqz-qwqx)],
       [2(qxqz-qwqy), 2(qyqz+qwqx), 1-2(qx²+qy²)]
   ]
   ```
4. Create 4×4 transformation matrix:
   ```
   T = [[R, t],
        [0, 1]]
   ```
   where `t = [x, y, z]`

**Output format**: 4×4 transformation matrix (homogeneous coordinates)

### Step 3: Temporal Synchronization

Since LiDAR and pose messages may have different timestamps, the script performs **nearest-neighbor matching**:

```python
for lidar_ts in lidar_timestamps:
    # Find nearest pose timestamp
    nearest_idx = np.argmin(np.abs(np.array(pose_timestamps) - lidar_ts))
    nearest_ts = pose_timestamps[nearest_idx]
    time_diff = abs(nearest_ts - lidar_ts)
    
    if time_diff > 0.1:  # Warn if > 100ms
        print(f"Warning: Large time difference ({time_diff:.3f}s)")
    
    poses_list.append(pose_data[nearest_ts])
```

**Process**:
1. Sort both LiDAR and pose timestamps
2. For each LiDAR frame, find the nearest pose timestamp
3. Warn if time difference exceeds 100ms
4. Assign the matched pose to the LiDAR frame

### Step 4: Data Saving

#### Point Cloud Files

```python
# Save point cloud as binary file (KITTI format: 4 floats per point)
filename = f"{i:06d}.bin"
filepath = os.path.join(velo_dir, filename)
points[:, :4].astype(np.float32).tofile(filepath)
```

**Format**: Binary file containing `float32` values
- **Structure**: Flat array of `[x, y, z, intensity]` repeated for each point
- **File naming**: `000000.bin`, `000001.bin`, `000002.bin`, ...
- **Location**: `{output_dir}/sequences/00/velodyne/`

#### Pose File

```python
# Save pose (4x4 matrix flattened to 12 values, last row is [0,0,0,1])
pose = poses_list[i]
pose_line = " ".join([str(pose[i, j]) for i in range(3) for j in range(4)])
f.write(pose_line + "\n")
```

**Format**: Text file with one line per frame
- **Structure**: 12 space-separated floats representing the first 3 rows of the 4×4 matrix
  - Row 1: `r11 r12 r13 tx`
  - Row 2: `r21 r22 r23 ty`
  - Row 3: `r31 r32 r33 tz`
  - Row 4: `[0, 0, 0, 1]` (implicit)
- **File**: `{output_dir}/sequences/00/poses.txt`

#### Calibration File

```python
# Identity transformation (can be adjusted if LiDAR-to-ego transform is known)
identity = np.eye(4)
f.write("Tr: ")
f.write(" ".join([str(identity[i, j]) for i in range(3) for j in range(4)]))
f.write("\n")
```

**Format**: Text file with calibration matrices
- **Structure**: `Key: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz`
- **Content**: `Tr` (transformation from LiDAR to ego vehicle frame)
- **Note**: For marine dataset, identity matrix is used (LiDAR frame = ego frame)
- **File**: `{output_dir}/sequences/00/calib.txt`

---

## Data Format Conversion

### KITTI-like Format Structure

The extracted data follows the KITTI dataset structure:

```
{output_dir}/
└── sequences/
    └── 00/
        ├── velodyne/          # Point cloud files
        │   ├── 000000.bin
        │   ├── 000001.bin
        │   └── ...
        ├── poses.txt          # Global poses (one per line)
        └── calib.txt          # Calibration matrices
```

### File Formats

1. **Point Clouds (`.bin` files)**:
   - Binary format: `float32` array
   - Shape: `(N, 4)` where N = number of points
   - Columns: `[x, y, z, intensity]`
   - Coordinates: LiDAR sensor frame

2. **Poses (`poses.txt`)**:
   - Text format: one line per frame
   - Each line: 12 floats (first 3 rows of 4×4 matrix)
   - Represents: Global pose of sensor platform

3. **Calibration (`calib.txt`)**:
   - Text format: `Key: values`
   - `Tr`: Transformation from LiDAR to ego frame (identity for marine dataset)

---

## Processing Pipeline

### Dataset Class: `data/marine.py`

The `MarineDataset` class handles data loading and preprocessing during training/testing.

### Step 1: Data Loading

```python
# Load point cloud
scan_path = os.path.join(self.marine_root, "sequences", sequence, "velodyne", velo_name)
scan = np.fromfile(scan_path, dtype=np.float32)
scan = scan.reshape((-1, 4))  # Reshape to (N, 4)

# Load pose
pose = self.poses[index]  # 4x4 transformation matrix
```

### Step 2: Ego Vehicle Filtering

**Purpose**: Remove points from the R/V Philos vessel (ego vehicle)

```python
def filter_out_ego(self, pts):
    """Filter out points from the ego vessel (R/V Philos)."""
    # R/V Philos dimensions:
    # - Length: ~7.6 m (x: -2.0 to 5.6 m)
    # - Width: ~2.7 m (y: -1.35 to 1.35 m)
    
    xx, yy, zz = pts[:, :3].T
    ego_mask = np.logical_and(
        np.logical_and(-1.35 <= yy, yy <= 1.35),  # Width
        np.logical_and(-2.0 <= xx, xx <= 5.6)     # Length
    )
    return pts[np.logical_not(ego_mask), :]
```

**Coordinate System**: KITTI frame (before transformation)
- **X-axis**: Forward (length of vessel)
- **Y-axis**: Left/right (width of vessel)
- **Z-axis**: Up/down (height)

**Filtering**: Points within bounding box `[-2.0, -1.35, -∞]` to `[5.6, 1.35, +∞]` are removed

### Step 3: Coordinate Transformation to Reference Frame

**Purpose**: Transform all points to a common reference frame (typically the reference frame at time `t=0`)

```python
# Reference frame's global pose
ref_pose = self.poses[ref_index]
inv_ref_pose = np.linalg.inv(ref_pose)

# Transform points to reference frame
pose = self.poses[index]  # Current frame's global pose
tf = inv_ref_pose @ pose  # Relative transformation
points_tf = (tf @ points.T).T  # Transform points
```

**Process**:
1. Get reference frame's global pose: `ref_pose`
2. Compute inverse: `inv_ref_pose = ref_pose⁻¹`
3. For each frame, compute relative transformation: `tf = inv_ref_pose @ pose`
4. Transform points: `points_tf = tf @ points`

**Result**: All points are in the reference frame's coordinate system

### Step 4: Coordinate System Conversion (KITTI → nuScenes)

**Purpose**: Convert from KITTI coordinate system to nuScenes coordinate system (used internally by the model)

```python
from data.common import KittiPoints2nuScenes

points_tf = KittiPoints2nuScenes(points_tf)
```

**Transformation**:
```python
def KittiPoints2nuScenes(points):
    # nuScenes x = - (KITTI y)
    # nuScenes y = (KITTI x)
    # nuScenes z = (KITTI z)
    xx, yy, zz = points[:, :3].T
    return np.stack((-yy, xx, zz)).T
```

**Coordinate System Comparison**:

| Axis | KITTI | nuScenes |
|------|-------|----------|
| X    | Forward | Right (Y in KITTI) |
| Y    | Left | Forward (X in KITTI) |
| Z    | Up | Up |

**Visual Representation**:
```
KITTI:          nuScenes:
   Y               X
   ↑               ↑
   |               |
   └──→ X          └──→ Y
   Z↑              Z↑
```

### Step 5: Range Filtering

**Purpose**: Remove points outside the specified spatial range

```python
def filter_by_range(self, pts):
    """Filter points within the specified pc_range (nuScenes coordinate system)."""
    xx, yy, zz = pts[:, :3].T
    x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
    x_mask = np.logical_and(x_min <= xx, xx < x_max)
    y_mask = np.logical_and(y_min <= yy, yy < y_max)
    z_mask = np.logical_and(z_min <= zz, zz < z_max)
    mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
    return pts[mask, :]
```

**Default Range** (from `configs/marine.yaml`):
```yaml
pc_range: [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
```
- **X**: -70.0 to 70.0 m (140 m total)
- **Y**: -70.0 to 70.0 m (140 m total)
- **Z**: -4.5 to 4.5 m (9 m total)

**Coordinate System**: nuScenes (after transformation)

### Step 6: Temporal Frame Selection

**Purpose**: Select input (past) and output (future) frames for temporal sequences

```python
# Calculate frame indices
first_index = ref_index - (self.n_input - 1) * self.input_step
last_index = ref_index + self.n_output * self.output_step

indices = [*range(first_index, ref_index + 1, self.input_step)] + \
          [*range(ref_index + self.output_step, last_index + 1, self.output_step)]
```

**Example** (with `n_input=5`, `input_step=2`, `n_output=5`, `output_step=2`, `ref_index=10`):
- **Input frames**: `[2, 4, 6, 8, 10]` (5 frames, step=2)
- **Output frames**: `[12, 14, 16, 18, 20]` (5 frames, step=2)

**Time Indices**:
- Input: `[4, 3, 2, 1, 0]` (0 = most recent, 4 = oldest)
- Output: `[0, 1, 2, 3, 4]` (0 = immediate future, 4 = farthest future)

---

## Coordinate System Transformations

### Transformation Chain

The data undergoes the following coordinate transformations:

1. **ROS LiDAR Frame** → **KITTI Frame** (during extraction)
   - Direct mapping (same convention)
   - X: forward, Y: left, Z: up

2. **KITTI Frame** → **Reference Frame** (during loading)
   - Transformation: `tf = inv_ref_pose @ pose`
   - All frames aligned to reference frame

3. **Reference Frame (KITTI)** → **nuScenes Frame** (during loading)
   - Transformation: `[x_nus, y_nus, z_nus] = [-y_kitti, x_kitti, z_kitti]`

4. **nuScenes Frame** → **Voxel Grid** (in model)
   - Discretization: `voxel_idx = (point - offset) / voxel_size`
   - Grid coordinates: `(t, h, l, w)` where `h, l, w` are height, length, width indices

### Coordinate System Details

#### KITTI Coordinate System
- **Origin**: LiDAR sensor center
- **X-axis**: Forward (vehicle direction)
- **Y-axis**: Left (perpendicular to forward)
- **Z-axis**: Up (vertical)

#### nuScenes Coordinate System
- **Origin**: LiDAR sensor center
- **X-axis**: Right (perpendicular to forward)
- **Y-axis**: Forward (vehicle direction)
- **Z-axis**: Up (vertical)

**Transformation Matrix**:
```
KITTI → nuScenes:
[0  -1  0]   [x_kitti]   [-y_kitti]
[1   0  0] × [y_kitti] = [x_kitti ]
[0   0  1]   [z_kitti]   [z_kitti ]
```

---

## Dataset Structure

### Directory Layout

```
{marine_root}/
└── sequences/
    └── 00/
        ├── velodyne/          # Point cloud files
        │   ├── 000000.bin
        │   ├── 000001.bin
        │   └── ...
        ├── poses.txt          # Global poses
        └── calib.txt          # Calibration
```

### Dataset Splits

Since the marine dataset is a single sequence, splits are **time-based** (frame-based):

```python
# Determine frame ranges
if marine_split == "train":
    frame_start, frame_end = 0, int(0.7 * total_frames)  # First 70%
elif marine_split == "val":
    frame_start, frame_end = int(0.7 * total_frames), int(0.85 * total_frames)  # Next 15%
elif marine_split == "test":
    frame_start, frame_end = int(0.85 * total_frames), total_frames  # Last 15%
```

**Split Percentages**:
- **Train**: 70% (first frames)
- **Validation**: 15% (middle frames)
- **Test**: 15% (last frames)
- **Trainval**: 85% (train + validation)

**Rationale**: Temporal splitting ensures that:
- Training uses earlier frames
- Testing uses later frames (future prediction)
- No data leakage between splits

---

## Training Data Flow

### Complete Pipeline

```
ROS Bag File
    ↓
[Extraction Script]
    ↓
KITTI-like Format
    ├── velodyne/*.bin (point clouds)
    ├── poses.txt (global poses)
    └── calib.txt (calibration)
    ↓
[MarineDataset.__getitem__]
    ├── Load point cloud (.bin file)
    ├── Load pose (from poses.txt)
    ├── Filter ego vehicle points
    ├── Transform to reference frame
    ├── Convert KITTI → nuScenes
    └── Filter by pc_range
    ↓
Model Input
    ├── input_points: (N_in, 3) - Past point clouds
    ├── input_tindex: (N_in,) - Time indices
    ├── output_origin: (n_output, 3) - Future frame origins
    ├── output_points: (N_out, 3) - Future point clouds (ground truth)
    └── output_tindex: (N_out,) - Time indices
```

### Data Preprocessing Summary

| Step | Input | Output | Coordinate System |
|------|-------|--------|-------------------|
| Extraction | ROS PointCloud2 | `.bin` files | LiDAR frame (KITTI) |
| Loading | `.bin` file | `(N, 4)` array | KITTI |
| Ego Filtering | `(N, 4)` array | `(M, 4)` array | KITTI |
| Reference Transform | `(M, 4)` array | `(M, 4)` array | Reference frame (KITTI) |
| nuScenes Conversion | `(M, 4)` array | `(M, 4)` array | nuScenes |
| Range Filtering | `(M, 4)` array | `(K, 4)` array | nuScenes |
| Model Input | `(K, 4)` array | `(K, 3)` tensor | nuScenes (voxel grid) |

**Note**: `N > M > K` (points are filtered at each step)

---

## Key Design Decisions

### 1. **KITTI-like Format**
- **Rationale**: Codebase originally designed for KITTI dataset
- **Benefit**: Minimal changes to existing code
- **Trade-off**: Requires format conversion from ROS

### 2. **Ego Vehicle Filtering**
- **Rationale**: Remove self-occlusions from stationary vessel
- **Benefit**: Focus on dynamic objects (passing ships)
- **Implementation**: Conservative bounding box based on vessel dimensions

### 3. **Temporal Splitting**
- **Rationale**: Single sequence dataset
- **Benefit**: No data leakage, realistic future prediction
- **Trade-off**: Smaller test set compared to multi-sequence datasets

### 4. **Coordinate System Conversion**
- **Rationale**: Model uses nuScenes coordinate system internally
- **Benefit**: Consistent with original model design
- **Implementation**: Simple axis swapping (KITTI Y → nuScenes X, KITTI X → nuScenes Y)

### 5. **Range Filtering**
- **Rationale**: Limit spatial extent for computational efficiency
- **Benefit**: Reduces memory usage, focuses on relevant area
- **Default**: 140m × 140m × 9m (suitable for marine environment)

---

## Usage Example

### Extraction Command

```bash
python scripts/extract_marine_data.py \
    --bag-path /path/to/section.bag \
    --output-dir /path/to/extracted/marine/data \
    --lidar-topic /velodyne_points \
    --pose-topic /lidar_odometry \
    --sequence-id 00 \
    --min-points 100
```

### Training Command

```bash
python train.py \
    --dataset marine \
    --marine-root /path/to/extracted/marine/data \
    --marine-cfg configs/marine.yaml \
    --n-input 5 --input-step 2 \
    --n-output 5 --output-step 2 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2
```

---

## Troubleshooting

### Common Issues

1. **Few points after filtering**:
   - Check `pc_range` is appropriate for your data
   - Verify ego filtering bounds match vessel dimensions
   - Inspect coordinate transformations

2. **Pose synchronization errors**:
   - Check time difference warnings in extraction log
   - Verify pose topic is correct
   - Consider interpolation if time differences are large

3. **Coordinate system mismatches**:
   - Verify KITTI → nuScenes transformation
   - Check that points are in expected range after transformation
   - Inspect visualization of transformed points

---

## References

- **Original Dataset**: MIT Sea Grant Philos dataset
- **ROS Message Types**: [sensor_msgs/PointCloud2](http://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html), [nav_msgs/Odometry](http://docs.ros.org/en/api/nav_msgs/html/msg/Odometry.html)
- **KITTI Dataset Format**: [KITTI Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- **Codebase**: Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting

---

*Last updated: January 2026*

