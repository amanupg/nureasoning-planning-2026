#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_fetch
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 2
#SBATCH --mem=3G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_fetch_%A_%a.out
# Alpamayo 2 Edge val/test frames (src/edge_fetch.py), 4 CPU shards. Re-running skips finished clips.
# Aman 11-11: CPU only, <= 4 shards at once, -c 2, memory from the measured MaxRSS of 13863759 (~0.88 GB/shard).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_HOME=/path/to/cluster/nureasoning/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
$D/.venv/bin/python -u src/edge_fetch.py --shard $SLURM_ARRAY_TASK_ID --nshards 4 --clips 4 --K 12 --splits ${SPLITS:-validation,test}
echo "EDGE_FETCH_MAIN_EXIT $?"
# Dup fix extras on the 31 val dup clips (frame K-21/-11/-6/-2 images; data/edge_members_valdup.json, own done marker;
# test members already include the fix). Test is dup-fixed by default (src/edge_inputs.py).
$D/.venv/bin/python -u src/edge_fetch.py --shard $SLURM_ARRAY_TASK_ID --nshards 4 --clips 4 --K 12 --splits validation \
  --members data/edge_members_valdup.json --done-name _done_dup.json
echo "EDGE_FETCH_EXIT $?"
