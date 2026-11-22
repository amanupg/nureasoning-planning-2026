#!/bin/bash
# Validation (scored) then test predictions for the 5 s-head runs h10 (frozen) and h10u4 (4 blocks unfrozen).
cd /path/to/workspace/jepa-nureasoning; PY=.venv/bin/python
export PYTHONPATH=/path/to/workspace/nureasoning/recon/devkit PYTHONDONTWRITEBYTECODE=1
for t in h10u4 h10; do $PY src/wajepa_ft.py predict --tag $t --split validation --score > logs/wajepa_ft_${t}_val.log 2>&1; done
for t in h10u4 h10; do $PY src/wajepa_ft.py predict --tag $t --split test > logs/wajepa_ft_${t}_test.log 2>&1; done
echo H10_PRED_DONE
