#!/bin/bash
# v8 pipeline (restart): WA-JEPA test -> v8.1; then validation runs and ensemble scoring for v8.3.
# Drive-JEPA test and v8.2 were done interactively (test images from the cluster copy, data/img_cl).
cd /path/to/workspace/jepa-nureasoning; PY=.venv/bin/python
wait_split() { until grep -q "\[img2\] $1 DONE" logs/ingest_v8.log; do sleep 60; done; }
wait_split test
$PY src/wajepa_run.py --split test --crop 169 > logs/wajepa_169_test.log 2>&1
$PY src/build_submission.py --name v8.1_wajepa-zeroshot_reasoning-v7.1 --members wa:1 > logs/build_v8.1.log 2>&1
wait_split validation
$PY src/drivejepa_run.py --split validation --score > logs/drivejepa_val.log 2>&1
$PY src/wajepa_run.py --split validation --crop 169 --score > logs/wajepa_169_val.log 2>&1
$PY src/ensemble_eval.py runs/wajepa_169_validation.npz runs/drivejepa_validation.npz > logs/ensemble_val.log 2>&1
echo PIPELINE_DONE
