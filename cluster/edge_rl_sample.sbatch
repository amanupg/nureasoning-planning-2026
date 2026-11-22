#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_rls
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-5
#SBATCH --exclude=ins058,ins082,ins095
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_rl_sample_%A_%a.out
# RL member, step 1 (src/edge_rl.py sample): K samples per selected TRAIN row (data/edge_rl_select.json) from MODEL,
# 6 shards x 1 GPU of any type (inference peak ~8.5 GiB), resumable (partial every 25 rows). Per-row noise seeds.
# Env vars: MODEL (required), TAG (e2a), K (6). Then on the Mac: edge_rl.py grade / build.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
N=/path/to/cluster/nureasoning
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
echo "== attempt $(date) host $(hostname)"
$N/ext/edge_env/bin/python -I -u src/edge_rl.py sample --model $MODEL --tag ${TAG:-e2a} --shard $SLURM_ARRAY_TASK_ID/6 --K ${K:-6}
echo "EDGE_RL_SAMPLE_EXIT $?"
