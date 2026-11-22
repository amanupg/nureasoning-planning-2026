#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_avla_sft
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095,ins046,ins047
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/avla_sft_%j.out
# AutoVLA fine-tuning probe (src/avla_sft.py). Blackwell nodes excluded (torch 2.4 pin). Env: TAG, N, ACCUM, LR, HOURS.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv-avla/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
$D/.venv-avla/bin/python -u src/avla_sft.py --tag ${TAG:-p1} --n ${N:-12000} --accum ${ACCUM:-8} --lr ${LR:-1e-4} --max-hours ${HOURS:-10.5}
