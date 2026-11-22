#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_rl
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins035,ins036,ins037,ins038,ins039,ins056,ins057,ins058,ins059,ins060,ins061,ins062,ins063,ins080,ins081,ins082,ins083,ins084,ins085,ins086,ins087,ins088,ins089,ins090,ins091,ins092,ins093,ins094,ins095
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_train_rl_%j.out
# RL member training (src/edge_train_rl.py) on one >= 80 GB GPU (same node exclusions as run B's widened jobs).
# Rows from src/edge_rl.py build: human-future anchor rows (normal CE + flow loss) plus positive-reward sample rows
# ("rl": 1, flow loss only, weight "w"). Init from MODEL (default e2-2400, the v9.9 model). Resumable like
# cluster/edge_train.sbatch: state saved hourly, planned stop 40 min before the limit (exit 3) -> requeue.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
N=/path/to/cluster/nureasoning
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== attempt $(date) host $(hostname) restarts ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
LIM=$(squeue -h -j $SLURM_JOB_ID -o %l); MIN=$(echo "$LIM" | awk -F'[-:]' '{if(NF==4)print $1*1440+$2*60+$3; else if(NF==3)print $1*60+$2; else print $1}')
$N/ext/edge_env/bin/python -I -u src/edge_train_rl.py --model ${MODEL:-weights/edge_ft_e2/update-2400} \
  --rows ${ROWS:-data/edge_rl_rows_e2a.json} --split train --out ${OUT:-weights/edge_rl_e2a} \
  --stop-after-min $((MIN - 40)) ${ARGS:-}; rc=$?
if [ $rc -eq 3 ]; then echo "requeue for resume"; scontrol requeue $SLURM_JOB_ID; fi
echo "EDGE_TRAIN_EXIT $rc"
