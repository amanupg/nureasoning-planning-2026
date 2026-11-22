#!/bin/bash
# B2 alone (restart with thread caps matching Insomnia's -c 8). Resumes from weights/edge_ft_b2/state.pt if present.
exec >> /root/pipe_a.log 2>&1
D=/path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
echo "== B2 (threads 8) $(date)"
$N/ext/edge_env/bin/python -I -u src/edge_train.py --model weights/edge_ft_b/update-3215 --rows data/edge_rows_train_r2.json --split train --out weights/edge_ft_b2 --seed 8 --lr 5e-6 --warmup 0.02 --save-every 800 --done-tag r2 >> logs/rp_b2.out 2>&1
echo "B2_EXIT $? $(date)"
