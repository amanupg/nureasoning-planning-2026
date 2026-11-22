#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_prof
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 0:40:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_prof_%j.out
# CPU-only profile of the Edge training input path (src/edge_prof.py).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
N=/path/to/cluster/nureasoning
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "host $(hostname)"; uptime
$N/ext/edge_env/bin/python -I -u src/edge_prof.py --model $N/models/alpamayo2_edge --rows data/edge_inputs_validation.json --split validation --n 12
echo "EDGE_PROF_EXIT $?"
