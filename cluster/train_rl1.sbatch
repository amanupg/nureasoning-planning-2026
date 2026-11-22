#!/bin/bash
#SBATCH -A edu
#SBATCH -p burst
#SBATCH -J jn_rl1
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/train_rl1_%j.out
# RL round 1: start from h10u4 and train toward its own high-NPS samples (runs/rl_h10u4_graded.npz, graded
# on the Mac with the devkit). Same architecture (5 s head, last 4 encoder blocks), lower lrs, 2 epochs.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok); sys.exit(0 if ok else 1)" || { echo 'CUDA NOT AVAILABLE - refusing to run on CPU'; exit 1; }
[ -f ${GRADED:-runs/rl_h10u4_graded.npz} ] || { echo 'graded samples missing'; exit 1; }
n=$(ls data/img_wa_full/train/*/_done.json 2>/dev/null | wc -l); ok=$(grep -l '"status": "ok"' data/img_wa_full/train/*/_done.json 2>/dev/null | wc -l)
echo "frames: $ok of 7654 clips ok ($n with a status file)"; [ "$ok" -eq 7654 ] || { echo "download incomplete - not training"; exit 1; }
$D/.venv/bin/python -u src/wajepa_ft.py train --tag ${TAG:-rl1} --seed 2 ${STEPS:+--steps $STEPS} --init-from ${INIT:-h10u4} --rl-graded ${GRADED:-runs/rl_h10u4_graded.npz} --rl-tau 0.05 --rl-min-best 0.5 --rl-gt ${GT:-0.2} --rl-mode ${MODE:-rwr} --rl-neg ${NEG:-0.25} --rl-cap ${CAP:-0.05} --rl-mix ${MIX:-0} --snapshots ${SNAP:-0} --no-resume ${NORESUME:-0} --img-root data/img_wa_full/train --crop full \
    --bs 4 --epochs ${EPOCHS:-2} --lr 2e-5 --enc-lr 5e-6 --wd 0.04 --workers 14 --dev-max ${EVAL:-2000} --eval-every ${EVERY:-2500} --horizon 10 --unfreeze-blocks 4 --log-every 200 --ckpt-minutes 30 &
PID=$!; trap 'kill -TERM $PID; wait $PID' TERM      # forward Slurm's pre-limit signal: save state, then exit
wait $PID
