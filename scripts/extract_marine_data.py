#!/usr/bin/env python3
"""
Extract LiDAR point clouds and pose data from ROS bag file for marine dataset.
Converts ROS bag data to KITTI-like format for use with the point cloud forecasting codebase.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Try to import rosbag (requires ROS installation)
ROS_AVAILABLE = False
try:
    import rosbag
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    import tf2_msgs.msg
    ROS_AVAILABLE = True
except ImportError:
    # Try bagpy as alternative (doesn't require full ROS installation)
    try:
        import bagpy
        from bagpy import bagreader
        ROS_AVAILABLE = False
        BAGPY_AVAILABLE = True
        print("Note: Using bagpy instead of rosbag. Some features may be limited.")
    except ImportError:
        print("Error: Neither rosbag nor bagpy is available.")
        print("Please install one of:")
        print("  1. ROS (for rosbag): Follow ROS installation instructions")
        print("  2. bagpy: pip install bagpy")
        print("     Note: bagpy may have limited support for PointCloud2 messages")
        sys.exit(1)


def extract_pointcloud(msg):
    """Extract point cloud from ROS PointCloud2 message."""
    points = []
    for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2], 1.0])  # x, y, z, intensity (default 1.0)
    return np.array(points, dtype=np.float32)


def pose_to_matrix(pose_msg):
    """Convert ROS pose message to 4x4 transformation matrix."""
    # Handle different message types
    if hasattr(pose_msg, 'pose'):
        # Odometry message
        if hasattr(pose_msg.pose, 'pose'):
            # PoseWithCovariance
            pose = pose_msg.pose.pose
        else:
            # Simple Pose
            pose = pose_msg.pose
    else:
        # Direct pose message
        pose = pose_msg
    
    # Extract position
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    
    # Extract orientation (quaternion)
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    
    # Convert quaternion to rotation matrix
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


def extract_marine_data(bag_path, output_dir, lidar_topic=None, pose_topic=None, 
                       sequence_id="00", min_points=100):
    """
    Extract LiDAR and pose data from ROS bag file.
    
    Args:
        bag_path: Path to ROS bag file
        output_dir: Output directory for extracted data
        lidar_topic: ROS topic name for LiDAR data (auto-detect if None)
        pose_topic: ROS topic name for pose/odometry data (auto-detect if None)
        sequence_id: Sequence identifier (default "00")
        min_points: Minimum points per frame to keep
    """
    print(f"Opening bag file: {bag_path}")
    bag = rosbag.Bag(bag_path, 'r')
    
    # Auto-detect topics if not provided
    topics = bag.get_type_and_topic_info()[1].keys()
    print(f"Available topics: {list(topics)}")
    
    if lidar_topic is None:
        # Try common LiDAR topic names
        lidar_candidates = ['/velodyne_points', '/lidar/points', '/points', 
                           '/velodyne/points', '/ouster/points', '/livox/lidar']
        lidar_topic = None
        for candidate in lidar_candidates:
            if candidate in topics:
                lidar_topic = candidate
                break
        if lidar_topic is None:
            # Find any PointCloud2 topic
            for topic in topics:
                if bag.get_type_and_topic_info()[1][topic].msg_type == 'sensor_msgs/PointCloud2':
                    lidar_topic = topic
                    break
    
    if pose_topic is None:
        # Try common pose/odometry topic names
        pose_candidates = ['/odom', '/imu/odom', '/pose', '/ego_pose', 
                          '/odometry/filtered', '/nav_msgs/Odometry']
        pose_topic = None
        for candidate in pose_candidates:
            if candidate in topics:
                pose_topic = candidate
                break
        if pose_topic is None:
            # Find any Odometry or PoseStamped topic
            for topic in topics:
                msg_type = bag.get_type_and_topic_info()[1][topic].msg_type
                if 'Odometry' in msg_type or 'PoseStamped' in msg_type:
                    pose_topic = topic
                    break
    
    if lidar_topic is None:
        raise ValueError("Could not find LiDAR topic. Please specify --lidar-topic")
    
    if pose_topic is None:
        print("Warning: Could not find pose topic. Will use identity poses.")
        print("You may need to extract poses separately or use IMU data.")
    
    print(f"Using LiDAR topic: {lidar_topic}")
    print(f"Using pose topic: {pose_topic}")
    
    # Create output directories
    seq_dir = os.path.join(output_dir, "sequences", sequence_id)
    velo_dir = os.path.join(seq_dir, "velodyne")
    os.makedirs(velo_dir, exist_ok=True)
    
    poses_file = os.path.join(seq_dir, "poses.txt")
    poses_list = []
    
    # Extract data
    lidar_timestamps = []
    lidar_data = {}
    pose_timestamps = []
    pose_data = {}
    
    print("Reading bag file...")
    for topic, msg, t in bag.read_messages(topics=[lidar_topic, pose_topic]):
        timestamp = t.to_sec()
        
        if topic == lidar_topic:
            if msg._type == 'sensor_msgs/PointCloud2':
                points = extract_pointcloud(msg)
                if len(points) >= min_points:
                    lidar_timestamps.append(timestamp)
                    lidar_data[timestamp] = points
        elif topic == pose_topic:
            # Handle different pose message types
            if 'Odometry' in msg._type or 'nav_msgs/Odometry' in msg._type:
                pose_matrix = pose_to_matrix(msg)
            elif 'PoseStamped' in msg._type or 'geometry_msgs/PoseStamped' in msg._type:
                pose_matrix = pose_to_matrix(msg)
            elif 'PoseWithCovariance' in msg._type:
                pose_matrix = pose_to_matrix(msg)
            else:
                print(f"Warning: Unknown pose message type: {msg._type}")
                continue
            pose_timestamps.append(timestamp)
            pose_data[timestamp] = pose_matrix
    
    bag.close()
    
    print(f"Found {len(lidar_timestamps)} LiDAR frames")
    print(f"Found {len(pose_timestamps)} pose frames")
    
    # Sort timestamps
    lidar_timestamps.sort()
    pose_timestamps.sort()
    
    # Match poses to LiDAR frames (nearest neighbor)
    if len(pose_timestamps) == 0:
        print("No pose data found. Using identity poses.")
        identity_pose = np.eye(4, dtype=np.float64)
        for i, lidar_ts in enumerate(lidar_timestamps):
            poses_list.append(identity_pose)
    else:
        for lidar_ts in lidar_timestamps:
            # Find nearest pose timestamp
            nearest_idx = np.argmin(np.abs(np.array(pose_timestamps) - lidar_ts))
            nearest_ts = pose_timestamps[nearest_idx]
            time_diff = abs(nearest_ts - lidar_ts)
            
            if time_diff > 0.1:  # Warn if time difference > 100ms
                print(f"Warning: Large time difference ({time_diff:.3f}s) between LiDAR and pose")
            
            poses_list.append(pose_data[nearest_ts])
    
    # Save point clouds and poses
    print("Saving extracted data...")
    with open(poses_file, 'w') as f:
        for i, (lidar_ts, points) in enumerate(zip(lidar_timestamps, lidar_data.values())):
            # Save point cloud as binary file (KITTI format: 4 floats per point)
            filename = f"{i:06d}.bin"
            filepath = os.path.join(velo_dir, filename)
            points[:, :4].astype(np.float32).tofile(filepath)
            
            # Save pose (4x4 matrix flattened to 12 values, last row is [0,0,0,1])
            pose = poses_list[i]
            pose_line = " ".join([str(pose[i, j]) for i in range(3) for j in range(4)])
            f.write(pose_line + "\n")
    
    # Create calibration file (identity transform for now)
    calib_file = os.path.join(seq_dir, "calib.txt")
    with open(calib_file, 'w') as f:
        # Identity transformation (can be adjusted if LiDAR-to-ego transform is known)
        identity = np.eye(4)
        f.write("Tr: ")
        f.write(" ".join([str(identity[i, j]) for i in range(3) for j in range(4)]))
        f.write("\n")
    
    print(f"Extraction complete!")
    print(f"Saved {len(lidar_timestamps)} frames to {seq_dir}")
    print(f"Point clouds: {velo_dir}")
    print(f"Poses: {poses_file}")
    print(f"Calibration: {calib_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LiDAR and pose data from ROS bag file")
    parser.add_argument("--bag-path", type=str, required=True,
                       help="Path to ROS bag file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for extracted data")
    parser.add_argument("--lidar-topic", type=str, default=None,
                       help="ROS topic name for LiDAR data (auto-detect if not specified)")
    parser.add_argument("--pose-topic", type=str, default=None,
                       help="ROS topic name for pose/odometry data (auto-detect if not specified)")
    parser.add_argument("--sequence-id", type=str, default="00",
                       help="Sequence identifier (default: 00)")
    parser.add_argument("--min-points", type=int, default=100,
                       help="Minimum points per frame to keep (default: 100)")
    
    args = parser.parse_args()
    
    extract_marine_data(
        args.bag_path,
        args.output_dir,
        args.lidar_topic,
        args.pose_topic,
        args.sequence_id,
        args.min_points
    )

