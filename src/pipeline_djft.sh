#!/bin/bash
# Drive-JEPA fine-tune: wait for the training-frame copy and the v8 GPU work (v8.1), then cache
# encoder features, train the decoder, and score on validation.
cd /path/to/workspace/jepa-nureasoning; PY=.venv/bin/python
export PYTHONPATH=/path/to/workspace/nureasoning/recon/devkit
until [ "$(find data/img_train -name '*.jpg' | wc -l | tr -d ' ')" -ge 81523 ] && ! pgrep -f "rsync -a --files-from=data/trainpart_" > /dev/null; do sleep 60; done
until grep -q "validate_submission\|Traceback" logs/build_v8.1.log 2>/dev/null; do sleep 60; done
$PY src/djft.py cache --split validation > logs/djft_cache_val.log 2>&1
$PY src/djft.py cache --split test > logs/djft_cache_test.log 2>&1
$PY src/djft.py cache --split train > logs/djft_cache_train.log 2>&1
$PY src/djft.py train --tag r40 > logs/djft_train_r40.log 2>&1
$PY src/djft.py predict --tag r40 --split validation --score > logs/djft_pred_r40_val.log 2>&1
echo DJFT_DONE
