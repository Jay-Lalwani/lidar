#!/bin/bash -l
# ===========================================================================
#  SLURM job script for 3D-Only Model Evaluation
#
#  Usage: sbatch scripts/evaluate_3d.slurm
# ===========================================================================
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J eval_3d
#SBATCH -o /p/cavalier/jay/logs/eval_3d/%A.out
#SBATCH -e /p/cavalier/jay/logs/eval_3d/%A.err
#SBATCH --export=NONE

set -euo pipefail

OPENPCDET_DIR="/p/cavalier/jay/OpenPCDet"
CKPT="/p/cavalier/jay/OpenPCDet/output/kitti_models/pointpillar/5458562/ckpt/checkpoint_epoch_80.pth"
CONFIG="cfgs/kitti_models/pointpillar.yaml"
SCORE_THRESH="${SCORE_THRESH:-0.1}"

echo "============================================"
echo "  3D-Only LiDAR Evaluation"
echo "  Job:  ${SLURM_JOB_ID:-LOCAL}  Node: $(hostname)"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Time: $(date)"
echo "  Checkpoint: ${CKPT}"
echo "  Score thresh: ${SCORE_THRESH}"
echo "============================================"

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
eval "$(conda shell.bash hook)"
conda activate /p/cavalier/jay/envs/openpcdet

# ── Create log directory ────────────────────────────────────────────────────
mkdir -p /p/cavalier/jay/logs/eval_3d

# ── Run evaluation ──────────────────────────────────────────────────────────
cd "${OPENPCDET_DIR}/tools"

python -u /p/cavalier/jay/scripts/evaluate_3d.py \
    --cfg_file "${CONFIG}" \
    --ckpt "${CKPT}" \
    --batch_size 4 \
    --workers 4 \
    --score_thresh "${SCORE_THRESH}" \
    --save_results /p/cavalier/jay/logs/eval_3d/results_${SLURM_JOB_ID:-LOCAL}.npz

echo ">>> Evaluation complete at $(date)"
