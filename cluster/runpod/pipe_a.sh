#!/bin/bash
# Pod A pipeline: F+other complete -> start r2 fetch + B1 -> r2 complete -> B2.
exec >> /root/pipe_a.log 2>&1
/root/waitpass.sh 9707 12 F other || exit 1
/root/fetch.sh 12 F:r2 other:r2
D=/path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$N/ext/edge_env/bin/python -I -u src/edge_train.py"
echo "== B1 resume $(date)"
$PY --model $N/models/alpamayo2_edge --rows data/edge_rows_train.json --split train --out weights/edge_ft_b --seed 7 --save-every 800 > logs/rp_b1.out 2>&1
rc=$?; echo "B1_EXIT $rc $(date)"; [ $rc -eq 0 ] && [ -d weights/edge_ft_b/update-3215 ] || exit 1
/root/waitpass.sh 9707 12 F:r2 other:r2 || exit 1
echo "== B2 $(date)"
$PY --model weights/edge_ft_b/update-3215 --rows data/edge_rows_train_r2.json --split train --out weights/edge_ft_b2 --seed 8 --lr 5e-6 --warmup 0.02 --save-every 800 --done-tag r2 > logs/rp_b2.out 2>&1
echo "B2_EXIT $? $(date)"
