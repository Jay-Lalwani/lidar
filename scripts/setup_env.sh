#!/bin/bash
#
# One-time environment setup for OpenPCDet PointPillars training.
#
# Run on a GPU node either interactively or via:
#   sbatch -p gpu --gres=gpu:1 -c 4 --mem=32G -t 01:00:00 \
#       -o /p/cavalier/jay/logs/setup_env-%A.out \
#       -e /p/cavalier/jay/logs/setup_env-%A.err \
#       setup_env.sh
#

echo "============================================"
echo "  OpenPCDet Environment Setup"
echo "============================================"

# --- Load modules ---
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1

echo "CUDA_HOME=${CUDA_HOME}"
which nvcc && nvcc --version

OPENPCDET_DIR=/p/cavalier/jay/OpenPCDet
ENV_PREFIX=/p/cavalier/jay/envs/openpcdet

# Use a writable package cache (system miniforge cache is read-only)
export CONDA_PKGS_DIRS=/p/cavalier/jay/envs/.conda_pkgs
mkdir -p "${CONDA_PKGS_DIRS}"

# --- Create conda environment ---
if [ -d "${ENV_PREFIX}" ] && [ -f "${ENV_PREFIX}/bin/python" ]; then
    echo "Conda environment already exists at ${ENV_PREFIX}. Activating..."
else
    echo "Creating conda environment at ${ENV_PREFIX} with Python 3.10..."
    mamba create --prefix "${ENV_PREFIX}" python=3.10 -y
fi

conda activate "${ENV_PREFIX}"

echo "Python: $(python --version)"
echo "Prefix: ${CONDA_PREFIX}"

# --- Install PyTorch via pip ---
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed, skipping..."
else
    echo ""
    echo "Installing PyTorch with CUDA 12.8 via pip..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# --- Install ninja (speeds up CUDA extension build) ---
pip install ninja

# --- Install OpenPCDet dependencies ---
echo ""
echo "Installing OpenPCDet Python dependencies..."
pip install numpy scipy easydict pyyaml scikit-image tqdm tensorboardX \
    numba SharedArray opencv-python pyquaternion

# --- Build OpenPCDet ---
echo ""
echo "Building OpenPCDet CUDA extensions..."
echo "CUDA_HOME=${CUDA_HOME}"
cd "${OPENPCDET_DIR}"
pip install -e . --no-build-isolation

# --- Verify ---
echo ""
echo "============================================"
echo "  Verifying installation..."
echo "============================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device:     {torch.cuda.get_device_name(0)}')
    print(f'CUDA version:    {torch.version.cuda}')

import pcdet
print(f'OpenPCDet:       OK')

# Check that CUDA extensions actually built
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
print(f'iou3d_nms_cuda:  OK')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with:"
echo "    module load miniforge/25.3.1-py3.12"
echo "    module load cuda/12.8.1"
echo "    conda activate ${ENV_PREFIX}"
echo "============================================"
