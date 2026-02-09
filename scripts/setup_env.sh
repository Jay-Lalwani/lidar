#!/bin/bash
# ===========================================================================
#  One-time environment setup for OpenPCDet PointPillars training on UVA HPC.
#
#  Creates a conda env, installs PyTorch + spconv + OpenPCDet, and builds
#  all CUDA extensions.  Must run on a GPU node (needs nvcc + GPU).
#
#  Usage:
#    sbatch -p gpu --gres=gpu:a100:1 -c 4 --mem=32G -t 01:00:00 \
#           -o /p/cavalier/jay/logs/setup-%A.out                  \
#           -e /p/cavalier/jay/logs/setup-%A.err                  \
#           scripts/setup_env.sh
# ===========================================================================
set -euo pipefail

ENV_PREFIX=/p/cavalier/jay/envs/openpcdet
OPENPCDET_DIR=/p/cavalier/jay/OpenPCDet

echo "============================================"
echo "  OpenPCDet Environment Setup"
echo "============================================"

# ── 1. Load system modules ──────────────────────────────────────────────────
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1

# Writable package cache (system miniforge cache is read-only)
export CONDA_PKGS_DIRS=/p/cavalier/jay/envs/.conda_pkgs
mkdir -p "${CONDA_PKGS_DIRS}"

# ── 2. Create / activate conda env ─────────────────────────────────────────
if [ -d "${ENV_PREFIX}" ] && [ -f "${ENV_PREFIX}/bin/python" ]; then
    echo "Conda env already exists at ${ENV_PREFIX}"
else
    echo "Creating conda env at ${ENV_PREFIX} (Python 3.10)..."
    mamba create --prefix "${ENV_PREFIX}" python=3.10 -y
fi
conda activate "${ENV_PREFIX}"
echo "Python: $(python --version)  Prefix: ${CONDA_PREFIX}"

# ── 3. Install PyTorch + CUDA ───────────────────────────────────────────────
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed, skipping."
else
    echo "Installing PyTorch (CUDA 12.8) via pip..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# ── 4. Install spconv (needed for voxelization in PointPillars) ─────────────
if python -c "from spconv.utils import Point2VoxelCPU3d" 2>/dev/null; then
    echo "spconv already installed, skipping."
else
    echo "Installing spconv-cu124..."
    pip install spconv-cu124
fi

# ── 5. Install remaining Python dependencies ────────────────────────────────
echo "Installing OpenPCDet Python dependencies..."
pip install ninja numpy scipy easydict pyyaml scikit-image tqdm tensorboardX \
    numba SharedArray opencv-python pyquaternion

# ── 6. Build OpenPCDet (CUDA extensions) ────────────────────────────────────
echo "Building OpenPCDet..."
cd "${OPENPCDET_DIR}"
pip install -e . --no-build-isolation

# ── 7. Verify everything ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
python -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'CUDA version:   {torch.version.cuda}')
import pcdet
print('OpenPCDet:      OK')
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
print('iou3d_nms_cuda: OK')
from spconv.utils import Point2VoxelCPU3d
print('spconv voxel:   OK')
"
echo ""
echo "Setup complete!"
