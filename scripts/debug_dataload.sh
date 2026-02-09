#!/bin/bash
module purge
module load miniforge/25.3.1-py3.12
module load cuda/12.8.1
conda activate /p/cavalier/jay/envs/openpcdet
cd /p/cavalier/jay/OpenPCDet/tools
python -u /p/cavalier/jay/scripts/debug_dataload.py
