#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_mem_b8
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/memtest_b8_%j.out
# Peak GPU memory and steady speed at batch 8 (and 6) on one high-memory GPU; 5 smoke clips (36 samples).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'available', ok); sys.exit(0 if ok else 1)" || { echo 'CUDA NOT AVAILABLE'; exit 1; }
for B in 8 6; do echo "== batch $B"
  $D/.venv/bin/python -u src/wajepa_ft.py train --tag mt_b$B --limit-clips 5 --dev-max 8 --log-every 1 --workers 6 --bs $B --steps 8 --eval-every 8 2>&1 | grep -v -i warn
done
rm -f runs/wajepa_ft_mt_b*
echo MEMTEST_DONE
