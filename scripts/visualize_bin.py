#!/usr/bin/env python3
"""
Visualize KITTI-format .bin point cloud files using Open3D.

This script loads and displays point cloud data from .bin files extracted
from the marine dataset (or any KITTI-format point cloud).
"""

import numpy as np
import open3d as o3d
import argparse
import os
import sys


def load_bin_file(bin_path):
    """
    Load point cloud from KITTI-format .bin file.
    
    Args:
        bin_path: Path to .bin file
        
    Returns:
        xyz: (N, 3) array of x, y, z coordinates
        intensity: (N,) array of intensity values (or None if not available)
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")
    
    # Load binary data
    points = np.fromfile(bin_path, dtype=np.float32)
    
    # Reshape to (N, 4) where columns are [x, y, z, intensity]
    if len(points) % 4 != 0:
        raise ValueError(f"File size ({len(points)} bytes) is not divisible by 4. "
                        f"Expected format: 4 floats per point (x, y, z, intensity)")
    
    points = points.reshape((-1, 4))
    
    # Extract coordinates and intensity
    xyz = points[:, :3]
    intensity = points[:, 3] if points.shape[1] >= 4 else None
    
    return xyz, intensity


def visualize_pointcloud(xyz, intensity=None, title="Point Cloud"):
    """
    Visualize point cloud using Open3D.
    
    Args:
        xyz: (N, 3) array of point coordinates
        intensity: (N,) array of intensity values (optional)
        title: Window title
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Color by intensity if available
    if intensity is not None:
        # Normalize intensity to [0, 1] for coloring
        intensity_min = intensity.min()
        intensity_max = intensity.max()
        
        if intensity_max > intensity_min:
            intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_norm = np.zeros_like(intensity)
        
        # Create color map (using viridis-like colors: blue -> green -> yellow -> red)
        colors = np.zeros((len(xyz), 3))
        colors[:, 0] = intensity_norm  # Red channel
        colors[:, 1] = intensity_norm * 0.8  # Green channel (slightly less)
        colors[:, 2] = 1.0 - intensity_norm * 0.5  # Blue channel (fades out)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"Intensity range: [{intensity_min:.2f}, {intensity_max:.2f}]")
    else:
        # Default color (white/gray)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Print statistics
    print(f"\nPoint Cloud Statistics:")
    print(f"  Number of points: {len(xyz):,}")
    print(f"  X range: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] m")
    print(f"  Y range: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] m")
    print(f"  Z range: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] m")
    
    # Visualize
    print(f"\nOpening visualization window. Close window to exit.")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI-format .bin point cloud files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a single file
  python visualize_bin.py /path/to/sequences/00/velodyne/000000.bin
  
  # Visualize with custom title
  python visualize_bin.py /path/to/sequences/00/velodyne/000000.bin --title "Frame 0"
        """
    )
    
    parser.add_argument(
        "bin_path",
        type=str,
        help="Path to .bin point cloud file"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Window title (default: filename)"
    )
    
    parser.add_argument(
        "--no-intensity",
        action="store_true",
        help="Don't color points by intensity (use uniform gray color)"
    )
    
    args = parser.parse_args()
    
    # Set title
    if args.title is None:
        title = os.path.basename(args.bin_path)
    else:
        title = args.title
    
    try:
        # Load point cloud
        print(f"Loading point cloud from: {args.bin_path}")
        xyz, intensity = load_bin_file(args.bin_path)
        
        # Visualize
        if args.no_intensity:
            visualize_pointcloud(xyz, intensity=None, title=title)
        else:
            visualize_pointcloud(xyz, intensity=intensity, title=title)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

