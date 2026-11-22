#!/bin/bash
#SBATCH -A edu
#SBATCH -p burst
#SBATCH -J jn_wa_w3
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/train_w3_%j.out
# WA-JEPA fine-tune, seed 2 (tag w3), 2 epochs (w1 peaked at ~1 epoch of 6): frozen V-JEPA 2.1 encoder; one GPU (any card on the cluster is >= 48 GB; peak 41 GiB at batch 6) (was: H100 80/94 GB or
# RTX PRO 6000), burst partition (no preemption, up to 14 days). Resumes from runs/wajepa_ft_w3_last.pt
# if present. Starts only when the frame download is complete (checked below as well).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok); sys.exit(0 if ok else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
n=$(ls data/img_wa_full/train/*/_done.json 2>/dev/null | wc -l); ok=$(grep -l '"status": "ok"' data/img_wa_full/train/*/_done.json 2>/dev/null | wc -l)
echo "frames: $ok of 7654 clips ok ($n with a status file)"; [ "$ok" -eq 7654 ] || { echo "download incomplete - not training"; exit 1; }
$D/.venv/bin/python -u src/wajepa_ft.py train --tag w3 --seed 2 --img-root data/img_wa_full/train --crop full \
    --bs 6 --epochs 2 --lr 6e-5 --wd 0.04 --workers 14 --dev-max 2000 --eval-every 3000 --log-every 200 --ckpt-minutes 30 &
PID=$!; trap 'kill -TERM $PID; wait $PID' TERM      # forward Slurm's pre-limit signal: save state, then exit
wait $PID
