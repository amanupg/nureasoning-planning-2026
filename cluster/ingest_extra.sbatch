#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_ingest_extra
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/ingest_extra_%A_%a.out
# B5 extra data: frames for the 2,236 train clips missing from the inherited index (data/members_wa_extra.json, 317,072 images),
# HTTP range reads from HF, stored as the model input (16:9 centre crop -> 256x512, JPEG q95; ~47 GB).
# 4 shards; re-running skips finished clips. HF token is READ from the main track's HF_HOME (no copy).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_HOME=/path/to/cluster/nureasoning/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export IMG_RESIZE=1408x928 IMG_WAJEPA=1
$D/.venv/bin/python -u src/img_ingest2.py --members data/members_wa_extra.json --splits train --out data/img_wa_full \
    --shard $SLURM_ARRAY_TASK_ID --nshards 4 --clips 8 --K 16
echo INGEST_SHARD_DONE
