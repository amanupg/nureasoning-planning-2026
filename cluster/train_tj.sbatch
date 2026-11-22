#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_tj
#SBATCH -N 1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084,ins092,ins095
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/train_tj_%j.out
# Loss-balance test (2026-10-28): src/grad_split.py showed the world-model (JEPA) loss supplies ~93% of the gradient
# in the shared predictor trunk, orthogonal to the planning gradient (cosine ~0). Continue from rl4 on the base data
# with the world-model weight JEPAW (0 = planning loss only); otherwise as fu2 (4 blocks, batch 4, gentle lrs).
# Dev FDE now uses fixed sampling noise. Preemptible (burst requeues with no grace time): resume file every 8 min; on short (12 h) since burst preempts every ~30 min,
# log appended across restarts. Env: TAG, INIT, JEPAW, LR, ENCLR, EPOCHS, EVERY, UNF.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE'; exit 1; }
$D/.venv/bin/python -u src/wajepa_ft.py train --tag ${TAG:-tj0} --seed 4 --init-from ${INIT:-rl4_s1987} --jepa-w ${JEPAW:-0} \
    --img-root data/img_wa_full/train --crop full --bs 4 --epochs ${EPOCHS:-1} --lr ${LR:-5e-6} --enc-lr ${ENCLR:-2e-6} --wd 0.04 \
    --workers 14 --dev-max 2000 --eval-every ${EVERY:-2500} --horizon 10 --unfreeze-blocks ${UNF:-4} --log-every 500 --ckpt-minutes 8
echo "== attempt end $(date)"; grep -q TRAINING_DONE <(tail -3 $D/logs/train_tj_${SLURM_JOB_ID}.out) && echo TRAIN_TJ_DONE
