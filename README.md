# PointPillars LiDAR Detection Pipeline

End-to-end pipeline for converting MCAP files to KITTI format and training a PointPillars model on the IAC racecar LiDAR data using OpenPCDet.

## Directory Structure

```
/p/cavalier/jay/
├── README.md                 ← This file
├── scripts/
│   ├── train.slurm           ← Unified training script
│   ├── evaluate.slurm        ← Rigorous evaluation script
│   ├── eval_rigorous.py      ← Evaluation metrics and logic
│   ├── validate_labels.py    ← Label visualization script
│   └── setup_env.sh          ← One-time environment setup script
├── OpenPCDet/                ← PointPillars training framework
│   ├── data/kitti/           ← The generated KITTI dataset
│   └── output/               ← Training checkpoints and logs
└── sandbox/
    ├── mcap_to_kitti.py      ← Universal unifier script
    └── run2_config.yaml      ← Data configuration map
```

## Step 1: Environment Setup

Run the setup script **ONE TIME** on a GPU node to construct the `openpcdet` conda environment:

```bash
mkdir -p /p/cavalier/jay/logs
sbatch scripts/setup_env.slurm
```

## Step 2: Training (and Data Export)

The master SLURM script handles both dataset export (from MCAP) and model training. 

```bash
sbatch scripts/train.slurm
```

**Inside `scripts/train.slurm` you can configure:**
* `REEXPORT_DATA=1/0`: Whether to run the `sandbox/mcap_to_kitti.py` export step first, repopulating `OpenPCDet/data/kitti/`.
* `VAL_RATIO`: Validation subsplit fraction (e.g. `0.2`).
* `SEED`: Random seed for identical dataset splitting.
* Standard SLURM parameters (`batch size`, `epochs`, `workers`, `ckpt_resume`).

If `REEXPORT_DATA=1`, intermediate frames without labels are ignored, and `val.txt` / `train.txt` sets are explicitly reconstructed with absolute disjoint separation.

## Step 3: Evaluation

To run rigorous evaluations on a trained checkpoint:

```bash
sbatch scripts/evaluate_3d.slurm
```

## MCAP-to-KITTI Export Logic Reference

When dataset exported is triggered, `mcap_to_kitti.py` processes MCAP data through a configuration map natively (bypassing old step-wise exports and interpolations).

**Example `run2_config.yaml`:**
```yaml
lidar_bag_dir: /path/to/rosbag2_*_lidar/    
camera_mcap: /path/to/front_left_center_compressed_0.mcap
ego_odom_mcap: /path/to/CavalierOdom/run.mcap
opponent_odom_mcaps:
  - /path/to/POLIMOVE/localized/run_0.mcap
  - /path/to/UNIMORE/localized/run_0.mcap
urdf: /path/to/av24.urdf
target_frame: center_of_gravity
val_ratio: 0.2
seed: 42
```
