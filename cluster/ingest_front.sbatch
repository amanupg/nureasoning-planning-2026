#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_ingest_front
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/ingest_front_%A_%a.out
# Distance model v1: FRONT camera at r and r-5 for the 81,523 training rows (163,046 frames, data/members_front_train.json),
# HTTP range reads from HF, stored at 1408x928 (full frame, no crop; ~38 GB, Aman OK 2026-10-21).
# 4 shards; re-running skips finished clips. HF token is READ from the main track's HF_HOME (no copy).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_HOME=/path/to/cluster/nureasoning/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export IMG_RESIZE=1408x928
$D/.venv/bin/python -u src/img_ingest2.py --members data/members_front_train.json --splits train --out data/img_front \
    --shard $SLURM_ARRAY_TASK_ID --nshards 4 --clips 8 --K 16
echo INGEST_SHARD_DONE
