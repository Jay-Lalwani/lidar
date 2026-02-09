#!/bin/bash
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
conda activate /p/cavalier/jay/envs/openpcdet
echo "Uninstalling spconv-cu120..."
pip uninstall spconv-cu120 cumm-cu120 -y
echo "Installing spconv-cu124..."
pip install spconv-cu124
echo "Verifying..."
python -c "
import cumm.tensorview as tv
import numpy as np
arr = np.random.randn(100, 4).astype(np.float32)
t = tv.from_numpy(arr)
print(f'tv.from_numpy: OK, shape={t.shape}')
from spconv.utils import Point2VoxelCPU3d
gen = Point2VoxelCPU3d(
    vsize_xyz=[0.16, 0.16, 4],
    coors_range_xyz=[-20.48, -39.68, -3, 199.68, 39.68, 1],
    num_point_features=4,
    max_num_points_per_voxel=32,
    max_num_voxels=40000)
out = gen.point_to_voxel(tv.from_numpy(arr))
print(f'Voxelization: OK, {out[0].shape[0]} voxels')
"
echo "Done!"
