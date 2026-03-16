#!/bin/bash -l
# ===========================================================================
#  SLURM job script for PointPillars Model Training & Data Generation.
#
#  Usage: sbatch scripts/train.slurm
#
#  Configurations available via environment variables:
#    REEXPORT_DATA=1 sbatch scripts/train.slurm
#    VAL_RATIO=0.2 sbatch scripts/train.slurm
# ===========================================================================
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J pp_train
#SBATCH -o /p/cavalier/jay/logs/train-%A.out
#SBATCH -e /p/cavalier/jay/logs/train-%A.err
#SBATCH --export=NONE

set -euo pipefail

# ── Configuration Parameters ────────────────────────────────────────────────
REEXPORT_DATA="${REEXPORT_DATA:-0}"      # Set to 1 to run mcap_to_kitti.py
VAL_RATIO="${VAL_RATIO:-0.2}"            # Train/validation split 
SEED="${SEED:-42}"                       # Random seed for data export
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-80}"
WORKERS="${WORKERS:-6}"

OPENPCDET_DIR="/p/cavalier/jay/OpenPCDet"
SANDBOX_DIR="/p/cavalier/jay/sandbox"
CONFIG_FILE="${SANDBOX_DIR}/run2_config.yaml"

echo "============================================"
echo "  PointPillars Training Pipeline"
echo "  Job:  ${SLURM_JOB_ID:-LOCAL}  Node: $(hostname)"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Time: $(date)"
echo "============================================"
echo "  REEXPORT_DATA: ${REEXPORT_DATA}"
echo "  VAL_RATIO:     ${VAL_RATIO}"
echo "  SEED:          ${SEED}"
echo "  BATCH_SIZE:    ${BATCH_SIZE}"
echo "============================================"

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
eval "$(conda shell.bash hook)"
conda activate /p/cavalier/jay/envs/openpcdet

# ── Step 1: Data Preprocessing ───────────────────────────────────────────────
if [ "${REEXPORT_DATA}" -eq 1 ]; then
    echo ">>> Running MCAP-to-KITTI Export with val_ratio=${VAL_RATIO}, seed=${SEED}..."
    cd "${SANDBOX_DIR}"
    
    # Update config file with fresh parameters and absolute target paths
    sed -i "s/^val_ratio:.*/val_ratio: ${VAL_RATIO}/" "${CONFIG_FILE}"
    sed -i "s/^seed:.*/seed: ${SEED}/" "${CONFIG_FILE}"
    sed -i "s|^output_dir:.*|output_dir: ${OPENPCDET_DIR}/data/kitti|" "${CONFIG_FILE}"
    
    # Execute the unified export directly parsing the ROS2 MCAP streams into pure KITTI format.
    python mcap_to_kitti.py --config "${CONFIG_FILE}"
    
    echo ">>> Export complete at $(date)"

cd "${OPENPCDET_DIR}"

echo ">>> Removing old data infos and GT database..."
rm -f data/kitti/kitti_infos_train.pkl
rm -f data/kitti/kitti_infos_val.pkl
rm -f data/kitti/kitti_infos_trainval.pkl
rm -f data/kitti/kitti_infos_test.pkl
rm -f data/kitti/kitti_dbinfos_train.pkl
rm -rf data/kitti/gt_database/

echo ">>> Regenerating KITTI data infos + GT database..."
python -u -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml
echo ">>> Info generation done at $(date)"

fi

# ── Step 2: Train Model ─────────────────────────────────────────────────────
cd "${OPENPCDET_DIR}/tools"

echo ">>> Training: batch=${BATCH_SIZE} epochs=${EPOCHS} workers=${WORKERS}"
python -u train.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --workers ${WORKERS} \
    --extra_tag ${SLURM_JOB_ID:-LOCAL}

echo ">>> Training complete at $(date)"
