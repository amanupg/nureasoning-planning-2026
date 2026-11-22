#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_plan_check
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/plan_fetch_check_%j.out
# Sample check of data/img_plan against the independent data/img_wa_full ingest + reasoning JSON parse (src/plan_fetch_check.py).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
PYTHONNOUSERSITE=1 $D/.venv/bin/python -u src/plan_fetch_check.py 30
echo CHECK_EXIT $?
