import os
import yaml

import numpy as np

import torch
from torch.utils.data import Dataset

from data.common import KittiPoint2nuScenes, KittiPoints2nuScenes


def parse_calibration(filename):
    """Read calibration file with given filename.
    
    Returns
    -------
    dict
        Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """Read poses file with per-scan poses from given filename.
    
    Returns
    -------
    list
        list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(Tr_inv @ (pose @ Tr))

    return poses


class MarineDataset(Dataset):
    """
    Dataset class for marine LiDAR data (MIT Sea Grant Philos dataset).
    Follows KITTI dataset structure for compatibility.
    """
    
    # For single sequence dataset, we can split by time/frame indices
    # Default: use first 70% for train, next 15% for val, last 15% for test
    SPLIT_SEQUENCES = {
        "train": ["00"],
        "val": ["00"],
        "test": ["00"],
        "trainval": ["00"],
        "all": ["00"]
    }
    
    # Frame indices for splits (will be set based on total frames)
    # These are relative indices within sequence "00"
    SPLIT_FRAME_RANGES = {
        "train": None,  # Will be set dynamically
        "val": None,
        "test": None,
        "trainval": None,
        "all": None
    }

    def __init__(self, marine_root, marine_cfg, marine_split, kwargs):
        """
        Initialize marine dataset.
        
        Args:
            marine_root: Root directory containing extracted marine data
            marine_cfg: Path to YAML config file (optional, can be None)
            marine_split: Split name ("train", "val", "test", "trainval", "all")
            kwargs: Dictionary with dataset parameters:
                - pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
                - voxel_size: Voxel size for discretization
                - n_input: Number of input frames
                - input_step: Step size for input frames
                - n_output: Number of output frames
                - output_step: Step size for output frames
        """
        # Setup pairs of input frames and output frames
        self.marine_root = marine_root
        if marine_cfg and os.path.exists(marine_cfg):
            self.info = yaml.safe_load(open(marine_cfg))
        else:
            self.info = {}

        self.pc_range = kwargs["pc_range"]
        self.voxel_size = kwargs["voxel_size"]

        self.n_input = kwargs["n_input"]
        self.input_step = kwargs.get("input_step", 1)
        self.n_output = kwargs["n_output"]
        self.output_step = kwargs.get("output_step", 1)

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.poses = []
        
        # Determine frame ranges for splits
        self.marine_split = marine_split

        for sequence in self.SPLIT_SEQUENCES[marine_split]:
            # Calibration file per sequence
            calib_path = os.path.join(marine_root, "sequences", sequence, "calib.txt")
            if not os.path.exists(calib_path):
                raise RuntimeError(f"Calibration file missing: {calib_path}")
            calib = parse_calibration(calib_path)

            # One pose file, many lines, one line per frame
            pose_path = os.path.join(marine_root, "sequences", sequence, "poses.txt")
            if not os.path.exists(pose_path):
                raise RuntimeError(f"Pose file missing: {pose_path}")
            poses = parse_poses(pose_path, calib)
            
            velo_dir = os.path.join(marine_root, "sequences", sequence, "velodyne")
            if not os.path.exists(velo_dir):
                raise RuntimeError("Velodyne directory missing: " + velo_dir)

            velo_names = sorted(os.listdir(velo_dir))
            total_frames = len(velo_names)
            
            # Determine frame ranges for this sequence
            if marine_split == "train":
                frame_start, frame_end = 0, int(0.7 * total_frames)
            elif marine_split == "val":
                frame_start, frame_end = int(0.7 * total_frames), int(0.85 * total_frames)
            elif marine_split == "test":
                frame_start, frame_end = int(0.85 * total_frames), total_frames
            elif marine_split == "trainval":
                frame_start, frame_end = 0, int(0.85 * total_frames)
            else:  # "all"
                frame_start, frame_end = 0, total_frames
            
            # Only add frames in the split range
            for idx in range(frame_start, frame_end):
                if idx < len(velo_names) and idx < len(poses):
                    self.sequences.append(sequence)
                    self.filenames.append(velo_names[idx])
                    self.poses.append(poses[idx])

        assert(len(self.sequences) == len(self.filenames) == len(self.poses))
        print(f"Marine dataset ({marine_split}): {len(self.filenames)} frames")

    def __len__(self):
        return len(self.filenames)

    def filter_by_range(self, pts):
        """Filter points within the specified pc_range (nuScenes coordinate system)."""
        xx, yy, zz = pts[:, :3].T
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        x_mask = np.logical_and(x_min <= xx, xx < x_max)
        y_mask = np.logical_and(y_min <= yy, yy < y_max)
        z_mask = np.logical_and(z_min <= zz, zz < z_max)
        mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
        return pts[mask, :]

    def filter_out_ego(self, pts):
        """
        Filter out points from the ego vessel (R/V Philos).
        
        R/V Philos dimensions:
        - 25 ft length (~7.6 m)
        - Width approximately 8-9 ft (~2.4-2.7 m)
        - Height approximately 6-7 ft (~1.8-2.1 m)
        
        Using conservative estimates in KITTI coordinate system:
        - x: forward (length) -2.0 to 5.6 m (7.6 m total)
        - y: left/right (width) -1.35 to 1.35 m (2.7 m total)
        - z: up/down (height) -1.0 to 1.0 m (2.0 m total, but less critical)
        """
        # KITTI coordinate system (will be transformed to nuScenes later)
        xx, yy, zz = pts[:, :3].T
        # Filter ego vessel: conservative bounding box
        ego_mask = np.logical_and(
            np.logical_and(-1.35 <= yy, yy <= 1.35),  # Width: ±1.35m
            np.logical_and(-2.0 <= xx, xx <= 5.6)     # Length: -2.0m to 5.6m
        )
        return pts[np.logical_not(ego_mask), :]

    def __getitem__(self, idx):
        """
        Get a data sample for training/testing.
        
        Returns:
            Tuple containing:
            - metadata: (sequence, filename, displacement)
            - input_points: Tensor of input point clouds
            - input_tindex: Tensor of input time indices
            - output_origin: Tensor of output frame origins
            - output_points: Tensor of output point clouds
            - output_tindex: Tensor of output time indices
        """
        # Do most of the heavy lifting (alignment across frames etc.)
        ref_index = idx

        ref_sequence = self.sequences[ref_index]
        ref_filename = self.filenames[ref_index]

        # Reference frame's global pose
        ref_pose = self.poses[ref_index]
        inv_ref_pose = np.linalg.inv(ref_pose)

        # Calculate frame indices for input and output
        first_index = ref_index - (self.n_input - 1) * self.input_step
        last_index = ref_index + self.n_output * self.output_step

        indices = [*range(first_index, ref_index + 1, self.input_step)] + \
            [*range(ref_index + self.output_step, last_index + 1, self.output_step)]

        input_points_list, input_tindex_list, input_origin_list = [], [], []
        output_points_list, output_tindex_list, output_origin_list = [], [], []
        
        for i, index in enumerate(indices):
            # Valid frame
            if 0 <= index and index < len(self.sequences) and self.sequences[index] == ref_sequence:
                sequence = self.sequences[index]
                velo_name = self.filenames[index]

                scan_path = os.path.join(self.marine_root, "sequences", sequence, "velodyne", velo_name)
                scan = np.fromfile(scan_path, dtype=np.float32)
                scan = scan.reshape((-1, 4))

                points = np.ones((scan.shape))
                points[:, :3] = scan[:, :3]

                # Remove returns from the ego vessel
                points = self.filter_out_ego(points)

                # Transform points to reference frame
                pose = self.poses[index]
                tf = inv_ref_pose @ pose
                origin_tf = tf[:3, 3].astype(np.float32)
                origin_tf = KittiPoint2nuScenes(origin_tf)

                # Transform points
                points_tf = (tf @ points.T).T
                points_tf = KittiPoints2nuScenes(points_tf)
                points_tf = points_tf.astype(np.float32)
                
                # Filter points within pc_range (nuScenes coordinate system)
                points_tf = self.filter_by_range(points_tf)

            else:
                # Invalid frame - use empty point cloud
                origin_tf = np.array([0, 0, 0], dtype=np.float32)
                points_tf = np.full((0, 3), float('nan'), dtype=np.float32)

            if i < self.n_input:
                tindex = np.full(len(points_tf), self.n_input - i - 1, dtype=np.float32)
                input_origin_list.append(origin_tf)
                input_points_list.append(points_tf)
                input_tindex_list.append(tindex)
            else:
                tindex = np.full(len(points_tf), i - self.n_input, dtype=np.float32)
                output_origin_list.append(origin_tf)
                output_points_list.append(points_tf)
                output_tindex_list.append(tindex)

        input_origin_tensor = torch.from_numpy(np.stack(input_origin_list))
        input_points_tensor = torch.from_numpy(np.concatenate(input_points_list))
        input_tindex_tensor = torch.from_numpy(np.concatenate(input_tindex_list))

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        output_points_tensor = torch.from_numpy(np.concatenate(output_points_list))
        output_tindex_tensor = torch.from_numpy(np.concatenate(output_tindex_list))

        displacement = torch.from_numpy(input_origin_list[0] - input_origin_list[1])

        return (ref_sequence, ref_filename, displacement), \
            input_points_tensor, input_tindex_tensor, \
            output_origin_tensor, output_points_tensor, output_tindex_tensor

