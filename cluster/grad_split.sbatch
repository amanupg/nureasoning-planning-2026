#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_grad
#SBATCH -N 1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084,ins092,ins095
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/grad_split_%j.out
# Loss-balance diagnostic (src/grad_split.py): 4 and 24 unfrozen blocks from rl4, 8 batches of 4 train rows.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for U in 4 24; do $D/.venv/bin/python -u src/grad_split.py --unfreeze $U --bs 2 2>&1 | grep -v Warning; done
