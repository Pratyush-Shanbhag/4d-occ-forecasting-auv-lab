import os
import json
import copy
import argparse
from datetime import datetime
import numpy as np
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import struct
from chamferdist import ChamferDistance
from data.common import CollateFn, nuScenesVolume2Kitti
from model import OccupancyForecastingNetwork
from utils.evaluation import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors, clamp
from plyfile import PlyData, PlyElement

from torch.utils.cpp_extension import load
import struct

dvr = load("dvr", sources=[
    "lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True)

def get_grid_mask(points, pc_range):
    points = points.T
    mask1 = np.logical_and(pc_range[0] <= points[0], points[0] <= pc_range[3])
    mask2 = np.logical_and(pc_range[1] <= points[1], points[1] <= pc_range[4])
    mask3 = np.logical_and(pc_range[2] <= points[2], points[2] <= pc_range[5])

    mask = mask1 & mask2 & mask3

    # print("shape of mask being returned", mask.shape)
    return mask

def get_rendered_pcds(origin, points, tindex, gt_dist, pred_dist, pc_range, eval_within_grid=False, eval_outside_grid=False):
    pcds = []
    for t in range(len(origin)):
        # FIX: Use pred_dist > 0.0 for predictions, not gt_dist > 0.0
        # This ensures we show all predicted points, not just where ground truth has valid rays
        mask = np.logical_and(tindex == t, pred_dist > 0.0)
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))
    return pcds

def get_clamped_output(origin, points, tindex, pc_range, gt_dist, eval_within_grid=False, eval_outside_grid=False, get_indices=False):
    pcds = []
    if get_indices:
        indices = []
    for t in range(len(origin)):
        mask = np.logical_and(tindex == t, gt_dist > 0.0)
        if eval_within_grid:
            mask = np.logical_and(mask, get_grid_mask(points, pc_range))
        if eval_outside_grid:
            mask = np.logical_and(mask, ~get_grid_mask(points, pc_range))
        # skip the ones with no data
        if not mask.any():
            continue
        if get_indices:
            idx = np.arange(points.shape[0])
            indices.append(idx[mask])
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        gt_pts = origin[t][None, :] + d * gt_dist[mask][:, None]
        pcds.append(torch.from_numpy(gt_pts))
    if get_indices:
        return pcds, indices
    else:
        return pcds

def make_data_loader(cfg, args):
    dataset_kwargs={
        "pc_range": cfg["pc_range"],
        "voxel_size": cfg["voxel_size"],
        "n_input": cfg["n_input"],
        "input_step": cfg["input_step"],
        "n_output": cfg["n_output"],
        "output_step": cfg["output_step"],
    }
    data_loader_kwargs={
        "pin_memory": False,  # NOTE
        "shuffle": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    if cfg["dataset"].lower() == "nuscenes":
        from data.nusc import nuScenesDataset
        from nuscenes.nuscenes import NuScenes
        from data.common import CollateFn

        if args.test_split == "test":
            cfg["nusc_version"] = "v1.0-test"
        nusc = NuScenes(cfg["nusc_version"], cfg["nusc_root"])

        Dataset = nuScenesDataset
        data_loader = DataLoader(
            Dataset(nusc, args.test_split, dataset_kwargs),
            collate_fn=CollateFn,
            **data_loader_kwargs,
        )
    elif cfg["dataset"].lower() == "kitti":
        from data.kitti import KittiDataset
        from data.common import CollateFn

        data_loader=DataLoader(
            KittiDataset(cfg["kitti_root"], cfg["kitti_cfg"], args.test_split, dataset_kwargs),
            collate_fn=CollateFn,
            **data_loader_kwargs,
        )
    elif cfg["dataset"].lower() == "argoverse2":
        from data.av2 import Argoverse2Dataset
        from data.common import CollateFn

        data_loader = DataLoader(
                Argoverse2Dataset(cfg["argo_root"], args.test_split, dataset_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            )
    elif cfg["dataset"].lower() == "marine":
        from data.marine import MarineDataset
        from data.common import CollateFn

        data_loader = DataLoader(
            MarineDataset(cfg["marine_root"], cfg.get("marine_cfg", "configs/marine.yaml"), args.test_split, dataset_kwargs),
            collate_fn=CollateFn,
            **data_loader_kwargs,
        )
    else:
        raise NotImplementedError(f"Dataset {cfg['dataset']} is not supported.")

    return data_loader


def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #
    device_count = torch.cuda.device_count()
    print("Device count", device_count)
    if args.batch_size % device_count != 0:
        raise RuntimeError(
            f"Batch size ({args.batch_size}) cannot be divided by device count ({device_count})"
        )

    #
    model_dir=args.model_dir
    with open(f"{model_dir}/config.json", "r") as f:
        cfg=json.load(f)
        if 'model_name' not in cfg:
            cfg['model_name'] = 'occ'

    if model_dir != cfg["model_dir"]:
        print("=" * 80)
        print(
            f"WARNING: inconsistent model directories: {model_dir} vs. {cfg['model_dir']}"
        )
        print("=" * 80)

    # dataset
    data_loader=make_data_loader(cfg, args)

    # instantiate a model and a renderer
    _n_input, _n_output=cfg["n_input"], cfg["n_output"]
    _pc_range, _voxel_size=cfg["pc_range"], cfg["voxel_size"]
    _model_type, _loss_type=cfg["model_type"], cfg["loss_type"]

    assert cfg["model_name"] == 'occ'
    model = OccupancyForecastingNetwork(
            _model_type, _loss_type, _n_input, _n_output, _pc_range, _voxel_size
        )

    # move onto gpu
    model=model.to(device)

    # resume
    ckpt_path=f"{args.model_dir}/ckpts/model_epoch_{args.test_epoch}.pth"
    checkpoint=torch.load(ckpt_path, map_location=device)
    # NOTE: ignore renderer's parameters
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # data parallel
    model=nn.DataParallel(model)
    model.eval()

    #
    dt=datetime.now()
    os.makedirs(f"{args.model_dir}/results/{args.test_split}/epoch_{args.test_epoch}", exist_ok=True)
    logfile = open(f"{args.model_dir}/results/{args.test_split}/epoch_{args.test_epoch}/{dt}.txt", "w")
    
    # Create directory for point clouds if saving is enabled
    if args.write_dense_pointcloud:
        pcd_output_dir = f"{args.model_dir}/results/{args.test_split}/epoch_{args.test_epoch}/pointclouds"
        os.makedirs(pcd_output_dir, exist_ok=True)
        print(f"Point clouds will be saved to: {pcd_output_dir}")

    metrics = {
        "count": 0.0,
        "chamfer_distance": 0.0,
        "chamfer_distance_inner": 0.0,
        "l1_error": 0.0,
        "absrel_error": 0.0
    }
    for i, batch in enumerate(data_loader):

        filenames=batch[0]

        input_points, input_tindex=batch[1:3]
        output_origin, output_points, output_tindex=batch[3:6]  # removed output_labels as
                                                                # the last returned argument

        if args.assume_constant_velocity:
                displacement = np.array([fname[-1].numpy() for fname in filenames]).reshape((-1, 1, 3))
                output_origin = torch.zeros_like(output_origin)
                displacements = torch.arange(_n_output) + 1
                output_origin = (output_origin + displacements[None, :, None]) * displacement

        if cfg["dataset"] == "nuscenes":
            output_labels = batch[6]
        else:
            output_labels = None

        bs=len(input_points)
        if bs % device_count != 0:
            print(f"Dropping the last batch of size {bs}")
            continue

        with torch.set_grad_enabled(False):
            ret_dict=model(
                input_points,
                input_tindex,
                output_origin,
                output_points,
                output_tindex,
                output_labels=output_labels,
                mode="testing",
                eval_within_grid=args.eval_within_grid,
                eval_outside_grid=args.eval_outside_grid
            )

        # iterate through the batch
        for j in range(output_points.shape[0]):  # iterate through the batch

            pred_pcds = get_rendered_pcds(
                    output_origin[j].cpu().numpy(),
                    output_points[j].cpu().numpy(),
                    output_tindex[j].cpu().numpy(),
                    ret_dict["gt_dist"][j].cpu().numpy(),
                    ret_dict["pred_dist"][j].cpu().numpy(),
                    _pc_range,
                    args.eval_within_grid,
                    args.eval_outside_grid
                )

            gt_pcds = get_clamped_output(
                    output_origin[j].cpu().numpy(),
                    output_points[j].cpu().numpy(),
                    output_tindex[j].cpu().numpy(),
                    _pc_range,
                    ret_dict["gt_dist"][j].cpu().numpy(),
                    args.eval_within_grid,
                    args.eval_outside_grid
                )

            # load predictions
            for k in range(len(gt_pcds)):
                pred_pcd = pred_pcds[k]
                gt_pcd = gt_pcds[k]
                origin = output_origin[j][k].cpu().numpy()

                # Statistics for this timestep
                pred_dist_t = ret_dict["pred_dist"][j][output_tindex[j] == k].cpu().numpy()
                gt_dist_t = ret_dict["gt_dist"][j][output_tindex[j] == k].cpu().numpy()
                
                # Log statistics
                if k == 0 and j == 0 and i == 0:  # Print header for first sample
                    print("\n" + "="*80)
                    print("PREDICTION STATISTICS")
                    print("="*80)
                
                if (i == 0 and j == 0) or (i % 10 == 0 and j == 0):  # Print for first batch or every 10 batches
                    print(f"\nBatch {i}, Sample {j}, Time {k}:")
                    print(f"  Pred points: {len(pred_pcd)}, GT points: {len(gt_pcd)}")
                    if np.any(pred_dist_t > 0):
                        print(f"  Pred dist - valid: {np.sum(pred_dist_t > 0)}/{len(pred_dist_t)}, "
                              f"mean: {np.mean(pred_dist_t[pred_dist_t > 0]):.2f}m")
                    else:
                        print(f"  Pred dist - valid: 0/{len(pred_dist_t)}, mean: N/A")
                    if np.any(gt_dist_t > 0):
                        print(f"  GT dist - valid: {np.sum(gt_dist_t > 0)}/{len(gt_dist_t)}, "
                              f"mean: {np.mean(gt_dist_t[gt_dist_t > 0]):.2f}m")
                    else:
                        print(f"  GT dist - valid: 0/{len(gt_dist_t)}, mean: N/A")
                    if len(pred_pcd) > 0 and len(gt_pcd) > 0:
                        print(f"  Pred range: X[{pred_pcd[:, 0].min():.2f}, {pred_pcd[:, 0].max():.2f}], "
                              f"Y[{pred_pcd[:, 1].min():.2f}, {pred_pcd[:, 1].max():.2f}], "
                              f"Z[{pred_pcd[:, 2].min():.2f}, {pred_pcd[:, 2].max():.2f}]")
                        print(f"  GT range: X[{gt_pcd[:, 0].min():.2f}, {gt_pcd[:, 0].max():.2f}], "
                              f"Y[{gt_pcd[:, 1].min():.2f}, {gt_pcd[:, 1].max():.2f}], "
                              f"Z[{gt_pcd[:, 2].min():.2f}, {gt_pcd[:, 2].max():.2f}]")

                # get the metrics
                metrics["count"] += 1
                metrics["chamfer_distance"] += compute_chamfer_distance(pred_pcd, gt_pcd, device)
                metrics["chamfer_distance_inner"] += compute_chamfer_distance_inner(pred_pcd, gt_pcd, device)
                l1_error, absrel_error = compute_ray_errors(pred_pcd, gt_pcd, torch.from_numpy(origin), device)
                metrics["l1_error"] += l1_error
                metrics["absrel_error"] += absrel_error
                
                # Save point clouds if requested
                if args.write_dense_pointcloud:
                    # Get sample identifier from filenames
                    if filenames[j]:
                        if isinstance(filenames[j], tuple) and len(filenames[j]) >= 2:
                            seq_id = str(filenames[j][0])
                            frame_id = str(filenames[j][1])
                        else:
                            seq_id = "00"
                            frame_id = f"{i:04d}_{j:04d}"
                    else:
                        seq_id = "00"
                        frame_id = f"{i:04d}_{j:04d}"
                    
                    # Convert to numpy
                    pred_np = pred_pcd.cpu().numpy() if isinstance(pred_pcd, torch.Tensor) else pred_pcd
                    gt_np = gt_pcd.cpu().numpy() if isinstance(gt_pcd, torch.Tensor) else gt_pcd
                    
                    # Ensure 3D points (N, 3)
                    if pred_np.shape[1] > 3:
                        pred_np = pred_np[:, :3]
                    if gt_np.shape[1] > 3:
                        gt_np = gt_np[:, :3]
                    
                    # Save predicted point cloud as PLY
                    pred_filename = f"{pcd_output_dir}/batch_{i:04d}_sample_{j:04d}_time_{k:02d}_pred.ply"
                    pred_vertex = np.array([tuple(p) for p in pred_np], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                    pred_el = PlyElement.describe(pred_vertex, 'vertex')
                    PlyData([pred_el]).write(pred_filename)
                    
                    # Save ground truth point cloud as PLY
                    gt_filename = f"{pcd_output_dir}/batch_{i:04d}_sample_{j:04d}_time_{k:02d}_gt.ply"
                    gt_vertex = np.array([tuple(p) for p in gt_np], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                    gt_el = PlyElement.describe(gt_vertex, 'vertex')
                    PlyData([gt_el]).write(gt_filename)

        print("Batch {"+str(i)+"/"+str(len(data_loader))+"}:", "Chamfer Distance:", metrics["chamfer_distance"] / metrics["count"])
        print("Batch {"+str(i)+"/"+str(len(data_loader))+"}:", "Chamfer Distance Inner:", metrics["chamfer_distance_inner"] / metrics["count"])
        print("Batch {"+str(i)+"/"+str(len(data_loader))+"}:", "L1 Error:", metrics["l1_error"] / metrics["count"])
        print("Batch {"+str(i)+"/"+str(len(data_loader))+"}:", "AbsRel Error:", metrics["absrel_error"] / metrics["count"])
        print("Batch {"+str(i)+"/"+str(len(data_loader))+"}:", "Count:", metrics["count"])

    print("Final Chamfer Distance:", metrics["chamfer_distance"] / metrics["count"])
    print("Final Chamfer Distance Inner:", metrics["chamfer_distance_inner"] / metrics["count"])
    print("Final L1 Error:", metrics["l1_error"] / metrics["count"])
    print("Final AbsRel Error:", metrics["absrel_error"] / metrics["count"])


    logfile.write("\nFinal Chamfer Distance: " + str(metrics["chamfer_distance"] / metrics["count"]))
    logfile.write("\nFinal Chamfer Distance Inner: " + str(metrics["chamfer_distance_inner"] / metrics["count"]))
    logfile.write("\nFinal L1 Error: " + str(metrics["l1_error"] / metrics["count"]))
    logfile.write("\nFinal AbsRel Error: " + str(metrics["absrel_error"] / metrics["count"]))

    logfile.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--test-split", type=str, required=True)
    parser.add_argument("--test-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--compute-chamfer-distance", action="store_true")
    parser.add_argument("--eval-within-grid", action="store_true")
    parser.add_argument("--eval-outside-grid", action="store_true")
    parser.add_argument("--assume-constant-velocity", action="store_true")
    parser.add_argument("--plot-metrics", action="store_true")
    parser.add_argument("--write-dense-pointcloud", action="store_true", 
                        help="Save predicted and ground truth point clouds as PLY files for visualization")

    args=parser.parse_args()
    torch.random.manual_seed(0)
    np.random.seed(0)
    test(args)
