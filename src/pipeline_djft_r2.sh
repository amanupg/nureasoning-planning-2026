#!/bin/bash
# Drive-JEPA fine-tune round 2 (pre-registered): r40 overfit after epoch 6 of a 40-epoch OneCycle run.
# Changes: 12-epoch OneCycle (the LR anneals by the point r40 began to overfit), weight decay 1e-4 -> 1e-2,
# dropout 0.0 -> 0.1 in the decoder transformer; same dev-log split as r40 (dev-seed 0); 3 seeds.
# Compare on validation: each seed and the 3-seed average vs r40 (0.6409). Then test predictions.
cd /path/to/workspace/jepa-nureasoning; PY=.venv/bin/python
export PYTHONPATH=/path/to/workspace/nureasoning/recon/devkit PYTHONDONTWRITEBYTECODE=1
for s in 0 1 2; do
  $PY src/djft.py train --tag r2s$s --seed $s --dev-seed 0 --epochs 12 --wd 0.01 --dropout 0.1 > logs/djft_train_r2s$s.log 2>&1
  $PY src/djft.py predict --tag r2s$s --split validation --score > logs/djft_pred_r2s${s}_val.log 2>&1
done
$PY src/djft.py avg --tags r2s0,r2s1,r2s2 --tag r2avg --split validation --score > logs/djft_avg_r2_val.log 2>&1
for s in 0 1 2; do $PY src/djft.py predict --tag r2s$s --split test > logs/djft_pred_r2s${s}_test.log 2>&1; done
$PY src/djft.py avg --tags r2s0,r2s1,r2s2 --tag r2avg --split test > logs/djft_avg_r2_test.log 2>&1
echo DJFT_R2_DONE
