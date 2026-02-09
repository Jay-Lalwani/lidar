#!/bin/bash
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
export CONDA_PKGS_DIRS=/p/cavalier/jay/envs/.conda_pkgs
conda activate /p/cavalier/jay/envs/openpcdet
echo "Installing spconv..."
pip install spconv-cu120
echo "Verifying..."
python -c "from spconv.utils import Point2VoxelCPU3d; print('spconv Point2VoxelCPU3d: OK')"
echo "Done!"
