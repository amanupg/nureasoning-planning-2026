#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_rl_sample
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/rl_sample_%A_%a.out
# RL round 1, step 1: 16 flow samples from h10u4 for 20,000 training frames (4 shards, resumable).
# Inference only (bf16), any GPU. LIMIT=<n> for a smoke test (first n frames of the shard).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
$D/.venv/bin/python -u src/rl_wajepa.py sample --tag ${SRC:-h10u4} --n ${N:-20000} --pool ${POOL:-train} --shard $SLURM_ARRAY_TASK_ID --nshards 4 --K ${K:-16} --chunk 16 \
    --img-root ${IMGROOT:-data/img_wa_full/train} --crop full --limit ${LIMIT:-0} --split ${SPLIT:-train} --save-scene ${SCENE:-0}
