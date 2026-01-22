#!/usr/bin/env python3
"""
Diagnostic script to analyze model predictions and identify issues.

This script loads a trained model and analyzes:
- pred_dist vs gt_dist statistics
- Number of valid rays
- Occupancy values in voxel grid
- Point cloud density
- Coordinate ranges
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OccupancyForecastingNetwork
from test import make_data_loader


def _get_range(tensor):
    """Get min/max range from tensor, handling NaN values."""
    valid = ~torch.isnan(tensor)
    if valid.any():
        return [float(tensor[valid].min().item()), float(tensor[valid].max().item())]
    else:
        return [0.0, 0.0]


def analyze_batch(model, batch, device, pc_range, voxel_size):
    """Analyze a single batch and return statistics."""
    filenames = batch[0]
    input_points, input_tindex = batch[1:3]
    output_origin, output_points, output_tindex = batch[3:6]
    
    # Move to device
    input_points = input_points.to(device)
    input_tindex = input_tindex.to(device)
    output_origin = output_origin.to(device)
    output_points = output_points.to(device)
    output_tindex = output_tindex.to(device)
    
    # Forward pass
    with torch.no_grad():
        ret_dict = model(
            input_points,
            input_tindex,
            output_origin,
            output_points,
            output_tindex,
            output_labels=None,
            mode="testing",
            eval_within_grid=False,
            eval_outside_grid=False
        )
    
    # Extract values
    pred_dist = ret_dict["pred_dist"].cpu().numpy()
    gt_dist = ret_dict["gt_dist"].cpu().numpy()
    sigma = ret_dict["sigma"].cpu().numpy()
    pog = ret_dict["pog"].cpu().numpy()
    
    stats = {
        "batch_size": len(input_points),
        "pred_dist": {
            "mean": float(np.mean(pred_dist[pred_dist > 0])),
            "std": float(np.std(pred_dist[pred_dist > 0])),
            "min": float(np.min(pred_dist[pred_dist > 0])) if np.any(pred_dist > 0) else 0.0,
            "max": float(np.max(pred_dist[pred_dist > 0])) if np.any(pred_dist > 0) else 0.0,
            "valid_count": int(np.sum(pred_dist > 0)),
            "total_count": int(pred_dist.size),
            "valid_ratio": float(np.sum(pred_dist > 0) / pred_dist.size) if pred_dist.size > 0 else 0.0,
        },
        "gt_dist": {
            "mean": float(np.mean(gt_dist[gt_dist > 0])),
            "std": float(np.std(gt_dist[gt_dist > 0])),
            "min": float(np.min(gt_dist[gt_dist > 0])) if np.any(gt_dist > 0) else 0.0,
            "max": float(np.max(gt_dist[gt_dist > 0])) if np.any(gt_dist > 0) else 0.0,
            "valid_count": int(np.sum(gt_dist > 0)),
            "total_count": int(gt_dist.size),
            "valid_ratio": float(np.sum(gt_dist > 0) / gt_dist.size) if gt_dist.size > 0 else 0.0,
        },
        "sigma": {
            "mean": float(np.mean(sigma)),
            "std": float(np.std(sigma)),
            "min": float(np.min(sigma)),
            "max": float(np.max(sigma)),
            "positive_count": int(np.sum(sigma > 0)),
            "total_count": int(sigma.size),
        },
        "pog": {
            "mean": float(np.mean(pog)),
            "std": float(np.std(pog)),
            "min": float(np.min(pog)),
            "max": float(np.max(pog)),
            "high_occ_count": int(np.sum(pog > 0.5)),  # High occupancy
            "total_count": int(pog.size),
        },
        "input_points": {
            "count": int(input_points.shape[1]) if len(input_points.shape) > 1 else 0,
            "valid_count": int(torch.sum(~torch.isnan(input_points[:, :, 0])).item()),
        },
        "output_points": {
            "count": int(output_points.shape[1]) if len(output_points.shape) > 1 else 0,
            "valid_count": int(torch.sum(~torch.isnan(output_points[:, :, 0])).item()),
        },
        "coordinate_ranges": {
            "input_x": _get_range(input_points[:, :, 0]) if input_points.shape[1] > 0 else [0, 0],
            "input_y": _get_range(input_points[:, :, 1]) if input_points.shape[1] > 0 else [0, 0],
            "input_z": _get_range(input_points[:, :, 2]) if input_points.shape[1] > 0 else [0, 0],
            "output_x": _get_range(output_points[:, :, 0]) if output_points.shape[1] > 0 else [0, 0],
            "output_y": _get_range(output_points[:, :, 1]) if output_points.shape[1] > 0 else [0, 0],
            "output_z": _get_range(output_points[:, :, 2]) if output_points.shape[1] > 0 else [0, 0],
        },
    }
    
    return stats


def diagnose(args):
    """Main diagnostic function."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    with open(args.model_dir + "/config.json", "r") as f:
        cfg = json.load(f)
    
    # Create data loader
    data_loader = make_data_loader(cfg, args)
    
    # Load model
    model = OccupancyForecastingNetwork(
        cfg["model_type"],
        cfg["loss_type"],
        cfg["n_input"],
        cfg["n_output"],
        cfg["pc_range"],
        cfg["voxel_size"],
    )
    
    # Load checkpoint
    ckpt_path = f"{args.model_dir}/ckpts/model_epoch_{args.test_epoch}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {ckpt_path}")
    print(f"Analyzing {len(data_loader)} batches...")
    
    # Collect statistics
    all_stats = []
    for i, batch in enumerate(data_loader):
        if i >= args.max_batches:
            break
        
        stats = analyze_batch(
            model, batch, device, cfg["pc_range"], cfg["voxel_size"]
        )
        stats["batch_idx"] = i
        all_stats.append(stats)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{min(len(data_loader), args.max_batches)} batches")
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    # Average statistics
    avg_pred_valid = np.mean([s["pred_dist"]["valid_ratio"] for s in all_stats])
    avg_gt_valid = np.mean([s["gt_dist"]["valid_ratio"] for s in all_stats])
    avg_pred_mean = np.mean([s["pred_dist"]["mean"] for s in all_stats if s["pred_dist"]["valid_count"] > 0])
    avg_gt_mean = np.mean([s["gt_dist"]["mean"] for s in all_stats if s["gt_dist"]["valid_count"] > 0])
    avg_sigma_mean = np.mean([s["sigma"]["mean"] for s in all_stats])
    avg_pog_mean = np.mean([s["pog"]["mean"] for s in all_stats])
    
    print(f"\nPrediction Distance (pred_dist):")
    print(f"  Valid rays ratio: {avg_pred_valid:.4f} ({avg_pred_valid*100:.2f}%)")
    print(f"  Mean distance (valid): {avg_pred_mean:.2f} m")
    
    print(f"\nGround Truth Distance (gt_dist):")
    print(f"  Valid rays ratio: {avg_gt_valid:.4f} ({avg_gt_valid*100:.2f}%)")
    print(f"  Mean distance (valid): {avg_gt_mean:.2f} m")
    
    print(f"\nOccupancy (sigma):")
    print(f"  Mean: {avg_sigma_mean:.4f}")
    print(f"  Positive count ratio: {np.mean([s['sigma']['positive_count']/s['sigma']['total_count'] for s in all_stats]):.4f}")
    
    print(f"\nProbability of Occupancy (pog):")
    print(f"  Mean: {avg_pog_mean:.4f}")
    print(f"  High occupancy ratio (>0.5): {np.mean([s['pog']['high_occ_count']/s['pog']['total_count'] for s in all_stats]):.4f}")
    
    print(f"\nPoint Cloud Statistics:")
    avg_input_valid = np.mean([s["input_points"]["valid_count"]/max(s["input_points"]["count"], 1) for s in all_stats])
    avg_output_valid = np.mean([s["output_points"]["valid_count"]/max(s["output_points"]["count"], 1) for s in all_stats])
    print(f"  Input points valid ratio: {avg_input_valid:.4f}")
    print(f"  Output points valid ratio: {avg_output_valid:.4f}")
    
    # Coordinate ranges
    print(f"\nCoordinate Ranges:")
    all_input_x = [s["coordinate_ranges"]["input_x"] for s in all_stats]
    all_output_x = [s["coordinate_ranges"]["output_x"] for s in all_stats]
    print(f"  Input X: [{min([r[0] for r in all_input_x]):.2f}, {max([r[1] for r in all_input_x]):.2f}]")
    print(f"  Output X: [{min([r[0] for r in all_output_x]):.2f}, {max([r[1] for r in all_output_x]):.2f}]")
    
    # Save detailed statistics
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nDetailed statistics saved to: {args.output_file}")
    
    # Issue detection
    print(f"\n" + "="*80)
    print("ISSUE DETECTION")
    print("="*80)
    
    issues = []
    if avg_pred_valid < 0.01:
        issues.append(f"WARNING: Very few valid prediction rays ({avg_pred_valid*100:.2f}%)")
    if avg_gt_valid < 0.01:
        issues.append(f"WARNING: Very few valid ground truth rays ({avg_gt_valid*100:.2f}%)")
    if avg_sigma_mean < 0.01:
        issues.append(f"WARNING: Very low occupancy values (mean sigma: {avg_sigma_mean:.4f})")
    if avg_pog_mean < 0.1:
        issues.append(f"WARNING: Very low probability of occupancy (mean: {avg_pog_mean:.4f})")
    if abs(avg_pred_mean - avg_gt_mean) > 10.0:
        issues.append(f"WARNING: Large difference between pred and gt distances ({abs(avg_pred_mean - avg_gt_mean):.2f}m)")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  No major issues detected")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose model predictions")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Model directory containing config.json and checkpoints")
    parser.add_argument("--test-epoch", type=int, default=14,
                       help="Epoch to test (default: 14)")
    parser.add_argument("--test-split", type=str, default="test",
                       help="Dataset split to use (default: test)")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for testing (default: 2)")
    parser.add_argument("--num-workers", type=int, default=2,
                       help="Number of workers (default: 2)")
    parser.add_argument("--max-batches", type=int, default=50,
                       help="Maximum number of batches to analyze (default: 50)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file for detailed statistics (JSON format)")
    
    args = parser.parse_args()
    
    diagnose(args)

