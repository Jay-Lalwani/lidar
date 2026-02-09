#!/bin/bash
# ===========================================================================
#  SLURM job script for PointPillars training on IAC racecar detection data.
#
#  First run:   sbatch scripts/train_pointpillars.slurm
#  Resume:      sbatch --export=RESUME=1 scripts/train_pointpillars.slurm
#
#  Prerequisites:
#    1. Run setup_env.sh to create the conda env + build OpenPCDet
#    2. Run preprocess_kitti.py to prepare the dataset
#    3. Data infos (pkl files) must exist — see SETUP.md step 5
# ===========================================================================
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J pointpillars_train
#SBATCH -o /p/cavalier/jay/logs/pp_train-%A.out
#SBATCH -e /p/cavalier/jay/logs/pp_train-%A.err

set -euo pipefail

OPENPCDET_DIR=/p/cavalier/jay/OpenPCDet
CKPT_DIR="${OPENPCDET_DIR}/output/kitti_models/pointpillar/default/ckpt"

BATCH_SIZE=16
EPOCHS=80
WORKERS=6

echo "============================================"
echo "  PointPillars Training"
echo "  Job:  ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Time: $(date)"
echo "============================================"

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
conda activate /p/cavalier/jay/envs/openpcdet

python -c "
import torch
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu} ({mem:.0f} GB)')
"

# ── Generate data infos (first run only) ────────────────────────────────────
cd "${OPENPCDET_DIR}"
if [ ! -f data/kitti/kitti_infos_train.pkl ]; then
    echo ">>> Generating KITTI data infos + GT database..."
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
        tools/cfgs/dataset_configs/kitti_dataset.yaml
    echo ">>> Done."
else
    echo ">>> Data infos already exist, skipping."
fi

# ── Resume logic ────────────────────────────────────────────────────────────
RESUME_ARGS=""
if [ "${RESUME:-0}" = "1" ] && [ -f "${CKPT_DIR}/latest_model.pth" ]; then
    echo ">>> Resuming from ${CKPT_DIR}/latest_model.pth"
    RESUME_ARGS="--ckpt ${CKPT_DIR}/latest_model.pth"
fi

# ── Train ───────────────────────────────────────────────────────────────────
# train.py resolves _BASE_CONFIG_ paths relative to CWD — must be tools/
cd "${OPENPCDET_DIR}/tools"

echo ">>> Training: batch=${BATCH_SIZE}  epochs=${EPOCHS}  workers=${WORKERS}"
python -u train.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --workers ${WORKERS} \
    ${RESUME_ARGS}

echo ">>> Training complete at $(date)"
