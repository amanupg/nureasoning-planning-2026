#!/bin/bash
# Score the WA-JEPA fine-tune (w1, best dev checkpoint = step 12000) on validation, then predict test.
cd /path/to/workspace/jepa-nureasoning; PY=.venv/bin/python
export PYTHONPATH=/path/to/workspace/nureasoning/recon/devkit PYTHONDONTWRITEBYTECODE=1
$PY src/wajepa_ft.py predict --tag w1 --split validation --score > logs/wajepa_ft_w1_val.log 2>&1
$PY src/wajepa_ft.py predict --tag w1 --split test > logs/wajepa_ft_w1_test.log 2>&1
echo W1_PRED_DONE
