#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_zs
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-1
#SBATCH --exclude=ins058,ins082,ins084
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_infer_%A_%a.out
# Alpamayo 2 Edge planning inference for a fine-tuned checkpoint (MODEL=...; src/edge_infer.py), 2 shards x 1 GPU, resumable (partial every 25 clips).
# Env and weights: the reasoning session's Edge env + model (read-only; repo pinned at 02f55a7, weights rev f8951d0).
# Env vars: INPUTS ('' or _dupfix for val; test is dup-fixed by default), SPLIT (validation), TAG (runs/edge_ft), K (6), LIMIT (0), NAV (override), TEMP (0.6), STEPS (10), SEED (43), NOTEXT (set: --no-text).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
N=/path/to/cluster/nureasoning
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
echo "== attempt $(date) host $(hostname)"
$N/ext/edge_env/bin/python -I -u src/edge_infer.py --model ${MODEL:-$N/models/alpamayo2_edge} --split ${SPLIT:-validation} \
  --out ${TAG:-runs/edge_ft} --shard $SLURM_ARRAY_TASK_ID/${NSH:-2} --K ${K:-6} --limit ${LIMIT:-0} --temperature ${TEMP:-0.6} --steps ${STEPS:-10} --seed ${SEED:-43} ${NAV:+--nav "$NAV"} ${INPUTS:+--inputs $INPUTS} ${NOTEXT:+--no-text}
echo "EDGE_INFER_EXIT $?"
