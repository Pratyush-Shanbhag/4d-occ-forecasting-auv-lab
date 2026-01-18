# Marine Dataset Extraction Script

## Usage

Extract LiDAR point clouds and pose data from ROS bag files for use with the point cloud forecasting codebase.

### Basic Usage

```bash
python scripts/extract_marine_data.py \
    --bag-path /path/to/section.bag \
    --output-dir /path/to/output/directory \
    --sequence-id 00
```

### Options

- `--bag-path`: Path to ROS bag file (required)
- `--output-dir`: Output directory for extracted data (required)
- `--lidar-topic`: ROS topic name for LiDAR data (optional, auto-detects if not specified)
- `--pose-topic`: ROS topic name for pose/odometry data (optional, auto-detects if not specified)
- `--sequence-id`: Sequence identifier (default: "00")
- `--min-points`: Minimum points per frame to keep (default: 100)

### Example

For the MIT Sea Grant Philos dataset:

```bash
python scripts/extract_marine_data.py \
    --bag-path /home/pratyush/ISyE_Research/datasets/unzipped/auv_lab/philos_2020_11_06_stationary_passed_by_launch/section.bag \
    --output-dir /home/pratyush/ISyE_Research/datasets/extracted/marine_philos \
    --sequence-id 00
```

### Output Structure

The script creates a KITTI-like directory structure:

```
output_dir/
└── sequences/
    └── 00/
        ├── velodyne/
        │   ├── 000000.bin
        │   ├── 000001.bin
        │   └── ...
        ├── poses.txt
        └── calib.txt
```

### Dependencies

Install ROS Python packages:

```bash
pip install rospkg sensor-msgs geometry-msgs nav-msgs tf2-msgs
```

Or if using ROS environment:

```bash
# Source ROS setup
source /opt/ros/<distro>/setup.bash
# Or use conda/venv with ROS packages installed
```

### Notes

- The script auto-detects common LiDAR topic names (`/velodyne_points`, `/lidar/points`, etc.)
- If pose topic is not found, identity poses will be used (you may need to extract poses separately)
- Point clouds are saved in binary format (4 floats per point: x, y, z, intensity)
- Poses are saved as 4x4 transformation matrices (12 values per line)

