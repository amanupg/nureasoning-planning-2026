#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_reas_drv
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/reasoning_driving_%j.out
# Driving blocks of the fetched train reasoning files -> data/reasoning_driving (src/reasoning_driving.py).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
PYTHONNOUSERSITE=1 $D/.venv/bin/python -u src/reasoning_driving.py 8
echo DRV_EXIT $?
