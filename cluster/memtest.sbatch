#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_memtest
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/memtest_%j.out
# WA-JEPA fine-tune smoke on ONE high-memory GPU (H100 80 GB or RTX PRO 6000; never the 48 GB cards):
# (1) real training steps at batch 1, (2) stop on a time limit and resume from the saved state,
# (3) peak memory at batch 2 and 4. Uses the 5 smoke clips already copied (data/img_wa/train).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok); sys.exit(0 if ok else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
PY=$D/.venv/bin/python; T="src/wajepa_ft.py train --limit-clips 5 --dev-max 4 --log-every 1 --workers 4"
echo "== run A: batch 1, stops itself after ~1 min"; $PY -u $T --tag mt1 --bs 1 --steps 40 --eval-every 20 --max-hours 0.015 2>&1 | grep -v -i warn
echo "== run B: same command, must resume"; $PY -u $T --tag mt1 --bs 1 --steps 40 --eval-every 20 2>&1 | grep -v -i warn
for B in 2 4; do echo "== batch $B"; $PY -u $T --tag mt_b$B --bs $B --steps 3 --eval-every 3 2>&1 | grep -v -i warn; done
rm -f runs/wajepa_ft_mt*.pt
echo MEMTEST_DONE
