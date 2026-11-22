#!/bin/bash
#SBATCH -A edu
#SBATCH -p burst
#SBATCH -J jn_full
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/train_full_%j.out
# B5 + full unfreeze (Aman 2026-10-24): WA-JEPA from rl4, all 24 encoder blocks trainable (batch 2: 30.6 GiB measured),
# base 7,654 clips + the 2,236 extra train clips (data/wa_train_meta_all.npz; dev logs unchanged).
# The extra clips' frames are staged to node-local /local (the shared edu fileset is full); the base frames are symlinked
# in, so one image root covers both. One long job, no resume file (6 GB); only the kept checkpoint goes to the fileset.
# Env: TAG, EPOCHS, INIT, LR, ENCLR, UNF (encoder blocks, default 24), BS (default 2), EVERY (dev eval, default 10000).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
$D/.venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'CUDA NOT AVAILABLE'; exit 1; }
LOC=/local/${USER}_jn_img; mkdir -p $LOC/train; df -h /local | tail -1
( export HF_HOME=/path/to/cluster/nureasoning/.cache/hf IMG_RESIZE=1408x928 IMG_WAJEPA=1
  $D/.venv/bin/python -u src/img_ingest2.py --members data/members_wa_extra.json --splits train --out $LOC --shard 0 --nshards 1 --clips 16 --K 16 ) 2>&1 | tail -3
ok=$(grep -l '"status": "ok"' $LOC/train/*/_done.json 2>/dev/null | wc -l); echo "extra clips staged: $ok of 2236"; [ "$ok" -ge 2230 ] || { echo "staging incomplete - not training"; exit 1; }
for c in data/img_wa_full/train/*/; do b=$(basename $c); [ -e $LOC/train/$b ] || ln -s $D/data/img_wa_full/train/$b $LOC/train/$b; done
echo "image root clips: $(ls $LOC/train | wc -l)"
$D/.venv/bin/python -u src/wajepa_ft.py train --tag ${TAG:-fu1} --seed 3 --init-from ${INIT:-rl4_s1987} --meta-tag all --img-root $LOC/train --crop full \
    --bs ${BS:-2} --epochs ${EPOCHS:-1} --lr ${LR:-2e-5} --enc-lr ${ENCLR:-1e-5} --wd 0.04 --workers 14 --dev-max 2000 --eval-every ${EVERY:-10000} --horizon 10 \
    --unfreeze-blocks ${UNF:-24} --log-every 500 --no-resume 1
echo TRAIN_FULL_DONE
