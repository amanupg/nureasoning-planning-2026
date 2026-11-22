#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_mem_u4
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 0:40:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/memtest_u4_%j.out
# Peak memory / speed: 5 s head with the last 4 encoder blocks trainable, batch 6 and 4, 5 smoke clips.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda', ok); sys.exit(0 if ok else 1)" || exit 1
for B in 6 4; do echo "== u4 h10 batch $B"
  $D/.venv/bin/python -u src/wajepa_ft.py train --tag mt_u4_b$B --limit-clips 5 --dev-max 6 --log-every 1 --workers 6 --bs $B --steps 6 --eval-every 6 --horizon 10 --unfreeze-blocks 4 2>&1 | grep -v -i warn
done
rm -f runs/wajepa_ft_mt_u4*
echo MEMTEST_DONE
