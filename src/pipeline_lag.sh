#!/bin/bash
# Sensor-sync test (nureasoning/LEDGER.md 2026-10-03: ego poses lead camera images by ~0.1 s):
# WA-JEPA zero-shot on validation with each pose paired with the camera frame one earlier (lag 1).
cd /path/to/workspace/jepa-nureasoning; PY=.venv/bin/python
export PYTHONPATH=/path/to/workspace/nureasoning/recon/devkit PYTHONDONTWRITEBYTECODE=1
until grep -q "\[img2\] validation DONE" logs/ingest_s1_val.log; do sleep 60; done
$PY src/wajepa_run.py --split validation --crop 169 --lag 1 --score > logs/wajepa_169_val_lag1.log 2>&1
echo LAG_DONE
