#!/bin/bash
#
# SLURM job script for training PointPillars on IAC racecar detection data.
#
# Usage:
#   sbatch train_pointpillars.slurm
#
# Prerequisites:
#   1. Run setup_env.sh on a GPU node to create the 'openpcdet' conda env
#   2. Run preprocess_kitti.py to prepare the dataset
#
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J pointpillars_train
#SBATCH -o /p/cavalier/jay/logs/pp_train-%A.out
#SBATCH -e /p/cavalier/jay/logs/pp_train-%A.err

set -euo pipefail

echo "============================================"
echo "  PointPillars Training Job"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Node:     $(hostname)"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Started:  $(date)"
echo "============================================"

# --- Environment setup ---
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
conda activate /p/cavalier/jay/envs/openpcdet

OPENPCDET_DIR=/p/cavalier/jay/OpenPCDet
cd "${OPENPCDET_DIR}"

# Quick sanity check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# --- Phase 1: Generate data infos + GT database (skip if already done) ---
if [ ! -f data/kitti/kitti_infos_train.pkl ]; then
    echo ""
    echo ">>> Generating KITTI data infos and GT database..."
    echo "    This may take a while on first run."
    echo ""
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
        tools/cfgs/dataset_configs/kitti_dataset.yaml
    echo ">>> Data info generation complete."
else
    echo ">>> KITTI data infos already exist, skipping generation."
fi

# --- Phase 2: Training ---
# train.py resolves _BASE_CONFIG_ relative to CWD, so run from tools/
cd "${OPENPCDET_DIR}/tools"

echo ""
echo ">>> Starting PointPillars training..."
echo "    Config: cfgs/kitti_models/pointpillar.yaml"
echo "    Batch size: 4"
echo "    Epochs: 80"
echo ""

python -u train.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --batch_size 4 \
    --epochs 80

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Finished: $(date)"
echo "============================================"
