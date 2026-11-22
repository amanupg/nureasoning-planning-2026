#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_dist
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/dist_%x_%j.out
# 5 s distance model (src/dist_net.py), DINOv3 ViT-B/16 fully fine-tuned. Env: TAG, EPOCHS, LIMIT, ROOT, HW ("256 512").
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv-qd/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
$D/.venv-qd/bin/python -u src/dist_net.py train --tag ${TAG:-d0} --epochs ${EPOCHS:-6} --root ${ROOT:-data/img_wa_full} --hw ${HW:-256 512} --bs ${BS:-32} --lr ${LR:-2e-5} --dec-w ${DECW:-0.5} ${LIMIT:+--limit $LIMIT}
