#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_ftr
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_fetch_train_%A_%a.out
# Edge fine-tuning frames for the 51,436 train rows (src/edge_rows.py): pass F (front camera, ~1056x696) then the
# other cameras (98304 px). 4 CPU shards (Aman 11-11), resumable (own done markers per pass). Saved tar indices from the first pass
# make the second pass cheap.
# Aman 11-11: CPU only, <= 4 shards at once, -c 2, memory from the measured MaxRSS of 13863759 (~0.88 GB/shard).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_HOME=/path/to/cluster/nureasoning/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
P="$D/.venv/bin/python -u src/edge_fetch.py --shard $SLURM_ARRAY_TASK_ID --nshards ${NSH:-4} --clips 4 --K 12 --splits train"
# TAG=r2 (run B's second frame draw): members data/edge_members_r2.json, markers _done_F_r2 / _done_other_r2.
[ -n "${TAG:-}" ] && P="$P --tag $TAG"
$P --cams F --maxpix 735000; echo "EDGE_FETCH_F_EXIT $?"
$P --cams other --maxpix 98304; echo "EDGE_FETCH_EXIT $?"
