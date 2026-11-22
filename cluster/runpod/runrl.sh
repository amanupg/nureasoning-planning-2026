#!/bin/bash
# RL member on a RunPod pod, same arguments as Insomnia job 13905289 (cluster/edge_train_rl.sbatch).
D=/path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning; cd $D
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
echo "== RL $(date) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$N/ext/edge_env/bin/python -I -u src/edge_train_rl.py --model weights/edge_ft_e2/update-2400 --rows data/edge_rl_rows_e2a.json \
  --split train --out weights/edge_rl_e2a --lr 2e-6 --warmup 0.05 --save-every 400 --seed 11 > logs/rp_rl.out 2>&1
echo "RL_EXIT $? $(date)"
