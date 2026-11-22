#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_memfull
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gres=gpu:R6000:1
#SBATCH -t 01:30:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/memtest_full_%j.out
# Full-encoder unfreeze memory test (24 blocks, from rl4): peak GPU memory at batch 4 and 2, 6 steps each, NO checkpoints.
# Run on a 96 GB card so the measurement is not cut off by an OOM.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
for BS in 4 2; do
  echo "== batch $BS"
  $D/.venv/bin/python -u src/wajepa_ft.py train --tag memfull$BS --seed 1 --init-from rl4_s1987 --img-root data/img_wa_full/train --crop full \
      --bs $BS --steps 6 --lr 2e-5 --enc-lr 1e-5 --wd 0.04 --workers 8 --dev-max 8 --eval-every 1000 --horizon 10 --unfreeze-blocks 24 \
      --log-every 1 --no-resume 1 --no-save 1 2>&1 | grep -E "trainable|step [0-9]/|Error|out of memory" | tail -8
done
echo MEMTEST_DONE
