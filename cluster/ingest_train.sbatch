#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_ingest
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/ingest_train_%A_%a.out
# WA-JEPA training frames: 958,344 camera members of 7,654 training clips (data/members_wa_train.json),
# HTTP range reads from HF, stored as the model input (16:9 centre crop -> 256x512, JPEG q95; ~47 GB).
# 4 shards; re-running skips finished clips. HF token is READ from the main track's HF_HOME (no copy).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_HOME=/path/to/cluster/nureasoning/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export IMG_RESIZE=1408x928 IMG_WAJEPA=1
$D/.venv/bin/python -u src/img_ingest2.py --members data/members_wa_train.json --splits train --out data/img_wa_full \
    --shard $SLURM_ARRAY_TASK_ID --nshards 4 --clips 8 --K 16
echo INGEST_SHARD_DONE
