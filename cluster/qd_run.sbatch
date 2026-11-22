#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_qd_run
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/qd_run_%A_%a.out
# Qwen-Drive-1.0-4B planning on nuReasoning scenes (src/qd_scenes.py). Resumable per shard.
# Env: PLANNER (planner-rl | planner-sft), MODE (reasoning_planning | direct_planning), NS (samples), SPLIT, LIMIT, TAG.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv-qd/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
SPLIT=${SPLIT:-validation}; TAG=${TAG:-rl}; mkdir -p runs/qd
$D/.venv-qd/bin/python -u ext/Qwen-Drive-1.0/scripts/run_planning.py --model weights/qwen-drive --planner weights/qwen-drive/${PLANNER:-planner-rl} \
    --scenes data/qd/${SPLIT}_scenes.jsonl --image-root ${IMG:-data/img_qd}/${SPLIT} --output runs/qd/${TAG}_${SPLIT}_s${SLURM_ARRAY_TASK_ID}.jsonl \
    --mode ${MODE:-reasoning_planning} --num-samples ${NS:-6} --num-shards ${NSH:-${SLURM_ARRAY_TASK_COUNT:-4}} --shard-index $SLURM_ARRAY_TASK_ID \
    --attn-implementation sdpa --num-workers 6 ${LIMIT:+--limit $LIMIT}
echo QD_RUN_DONE
