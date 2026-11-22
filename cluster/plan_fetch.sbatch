#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_plan_fetch
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --array=0-5
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/plan_fetch_%A_%a.out
# Planning images (3 front cams x 4 frames per row) + missing train reasoning files for the 10B planning recipe
# (data/plan_members.json from src/plan_members.py). HTTP range reads from the HF tars, resized in memory to
# 546x359-class, JPEG q95, written to data/img_plan/<split>/... Val and test first, then train. 6 shards; re-running
# skips finished clips; every process stops itself when all runs together have written > CAP_GB (default 48; Aman approved 56 on 11-11).
# HF token is READ from the main track's HF_HOME (600 file, no copy, never argv).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_HOME=/path/to/cluster/nureasoning/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# PHASE=A: val + test images, then the missing train reasoning files (rows must be rebuilt from them before the train
# images, because new decision frames move the regular frames). PHASE=B: train images (from the rebuilt member list).
P="$D/.venv/bin/python -u src/plan_fetch.py --shard $SLURM_ARRAY_TASK_ID --nshards 6 --clips 4 --K 24 --cap-gb ${CAP_GB:-48}"
if [ "${PHASE:-A}" = A ]; then
  $P --splits validation,test --mode images ${PLAN_FETCH_ARGS:-}; echo "PLAN_FETCH_EXIT_VALTEST $?"
  $P --splits train --mode reasoning ${PLAN_FETCH_ARGS:-}; echo "PLAN_FETCH_EXIT $?"
elif [ "${PHASE}" = DUP ]; then
  # duplicate-frame fix: val/test images one frame earlier (data/plan_members_dup.json; plus the 2 test clips whose
  # saved index was stale), own done marker so the finished val/test clips are not skipped
  $P --splits validation,test --mode images --members data/plan_members_dup.json --done-name _done_dup.json ${PLAN_FETCH_ARGS:-}; echo "PLAN_FETCH_EXIT $?"
else
  $P --splits train --mode images ${PLAN_FETCH_ARGS:-}; echo "PLAN_FETCH_EXIT $?"
fi
