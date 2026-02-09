#!/usr/bin/env python3
"""Quick diagnostic to isolate the SIGFPE in the data pipeline.
Run from OpenPCDet/tools/ directory."""
import faulthandler
faulthandler.enable()
import sys, os, numpy as np
from pathlib import Path
from easydict import EasyDict
import yaml

print("=== Step 1: Load config ===")
with open("cfgs/kitti_models/pointpillar.yaml", 'r') as f:
    model_cfg = EasyDict(yaml.safe_load(f))
with open(model_cfg.DATA_CONFIG._BASE_CONFIG_, 'r') as f:
    dataset_cfg = EasyDict(yaml.safe_load(f))
for key in model_cfg.DATA_CONFIG:
    if key != '_BASE_CONFIG_':
        dataset_cfg[key] = model_cfg.DATA_CONFIG[key]
print(f"  POINT_CLOUD_RANGE: {dataset_cfg.POINT_CLOUD_RANGE}")

print("\n=== Step 2: Load a sample point cloud ===")
data_root = Path("../data/kitti")
with open(data_root / "ImageSets" / "train.txt") as f:
    sample_ids = [l.strip() for l in f.readlines()]
sid = sample_ids[0]
print(f"  Sample: {sid}")
pc_path = data_root / "training" / "velodyne" / f"{sid}.bin"
points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
print(f"  Points: {points.shape}, NaN={np.any(np.isnan(points))}, Inf={np.any(np.isinf(points))}")
print(f"  X=[{points[:,0].min():.1f},{points[:,0].max():.1f}] Y=[{points[:,1].min():.1f},{points[:,1].max():.1f}] Z=[{points[:,2].min():.1f},{points[:,2].max():.1f}]")

print("\n=== Step 3: Test voxelization ===")
pc_range = np.array(dataset_cfg.POINT_CLOUD_RANGE)
mask = ((points[:,0]>=pc_range[0])&(points[:,0]<=pc_range[3])&
        (points[:,1]>=pc_range[1])&(points[:,1]<=pc_range[4])&
        (points[:,2]>=pc_range[2])&(points[:,2]<=pc_range[5]))
pts = points[mask]
print(f"  After range mask: {pts.shape[0]} pts")
vs = dataset_cfg.DATA_PROCESSOR[2].VOXEL_SIZE
grid = ((pc_range[3:6]-pc_range[0:3])/np.array(vs)).astype(np.int64)
print(f"  voxel_size={vs}, grid={grid}")
try:
    import cumm.tensorview as tv
    from spconv.utils import Point2VoxelCPU3d
    gen = Point2VoxelCPU3d(vsize_xyz=vs, coors_range_xyz=dataset_cfg.POINT_CLOUD_RANGE,
        num_point_features=4, max_num_points_per_voxel=32, max_num_voxels=40000)
    out = gen.point_to_voxel(tv.from_numpy(pts))
    print(f"  OK: {out[0].shape[0]} voxels")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== Step 4: Full dataset (no aug) ===")
try:
    dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'gt_sampling','random_world_flip','random_world_rotation','random_world_scaling']
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    ds = KittiDataset(dataset_cfg=dataset_cfg, class_names=['Car'],
        root_path=data_root.resolve(), training=True)
    print(f"  {len(ds)} samples, loading [0]...")
    item = ds[0]
    print(f"  OK: keys={list(item.keys())}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Step 5: Full dataset (with aug) ===")
try:
    dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['placeholder']
    ds2 = KittiDataset(dataset_cfg=dataset_cfg, class_names=['Car'],
        root_path=data_root.resolve(), training=True)
    print(f"  Loading [0] with augmentation...")
    item2 = ds2[0]
    print(f"  OK: keys={list(item2.keys())}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n=== Done ===")
