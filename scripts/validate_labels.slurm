#!/bin/bash
# ===========================================================================
#  SLURM job script for PointPillars label validation/visualization.
#
#  Run:   sbatch scripts/validate_labels.slurm
#
#  This script runs validate_labels.py to generate interactive HTML visualizations
#  of the dataset labels (and optional model predictions).
# ===========================================================================
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J pp_validate
#SBATCH -o /p/cavalier/jay/logs/validate-%A.out
#SBATCH -e /p/cavalier/jay/logs/validate-%A.err

set -euo pipefail

OPENPCDET_DIR=/p/cavalier/jay/OpenPCDet
CKPT="${OPENPCDET_DIR}/output/kitti_models/pointpillar/5458562/ckpt/checkpoint_epoch_80.pth"
# CKPT=""

SAMPLES=10
OUT_DIR="/p/cavalier/jay/logs/vis/${SLURM_JOB_ID}"
# Split to visualize: 'train' or 'val'
SPLIT="val"

echo "============================================"
echo "  PointPillars Label Validation"
echo "  Job:  ${SLURM_JOB_ID}  Node: $(hostname)"
echo "  Time: $(date)"
echo "============================================"

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
eval "$(conda shell.bash hook)"
conda activate /p/cavalier/jay/envs/openpcdet

# ── Validation ──────────────────────────────────────────────────────────────
# validate_labels.py should be run from tools/ so that default config paths resolve correctly.
cd "${OPENPCDET_DIR}/tools"

CMD_ARGS="--samples ${SAMPLES} --out ${OUT_DIR} --split ${SPLIT}"

if [ -n "${INDICES:-}" ]; then
    CMD_ARGS="${CMD_ARGS} --indices ${INDICES}"
fi

if [ -n "${CKPT:-}" ]; then
    CMD_ARGS="${CMD_ARGS} --ckpt ${CKPT}"
fi

echo ">>> Running validation..."
echo ">>> Command: python /p/cavalier/jay/scripts/validate_labels.py ${CMD_ARGS}"

python /p/cavalier/jay/scripts/validate_labels.py ${CMD_ARGS}

echo ">>> Validation complete at $(date)"
echo ">>> Results in: ${OUT_DIR}"
