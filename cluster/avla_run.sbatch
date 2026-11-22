#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_avla
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095,ins046,ins047
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH --array=0-3
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/avla_%A_%a.out
# AutoVLA inference (src/avla_run.py), resumable per shard. ins046-047 excluded: Blackwell sm_120 is unsupported by its torch 2.4 pin. Env: SPLIT, LIMIT, TAG, CKPT, TEMP.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv-avla/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
$D/.venv-avla/bin/python -u src/avla_run.py --split ${SPLIT:-validation} --shard $SLURM_ARRAY_TASK_ID --nshards ${NSH:-${SLURM_ARRAY_TASK_COUNT:-4}} \
    --tag ${TAG:-zs} --ckpt ${CKPT:-weights/autovla/AutoVLA_PDMS_89.ckpt} --temperature ${TEMP:-0.01} ${LIMIT:+--limit $LIMIT}
