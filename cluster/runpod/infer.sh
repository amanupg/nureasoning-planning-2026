#!/bin/bash
# infer.sh WAITLOG WAITPAT MODEL TAG SEED : wait for WAITPAT in WAITLOG, then run val (dupfix) and test inference
# concurrently on this pod's GPU (K=6, T=0.6), thread caps for the 13.6-CPU quota. Writes INFER_DONE/INFER_FAIL.
WL=$1; WP=$2; M=$3; T=$4; SEED=$5
D=/path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning; cd $D
exec >> /root/infer_$(basename $T).log 2>&1
until grep -q "$WP" $WL 2>/dev/null; do grep -q -E "_EXIT [1-9]" $WL 2>/dev/null && { echo "INFER_FAIL upstream $(date)"; exit 1; }; sleep 30; done
[ -d $M ] || { echo "INFER_FAIL no model $M $(date)"; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "== infer $M tag $T seed $SEED $(date)"
$N/ext/edge_env/bin/python -I -u src/edge_infer.py --model $M --split validation --inputs _dupfix --out $T --shard 0/1 --K 6 --temperature 0.6 --seed $SEED > logs/infer_$(basename $T)_val.out 2>&1 & p1=$!
$N/ext/edge_env/bin/python -I -u src/edge_infer.py --model $M --split test --out $T --shard 0/1 --K 6 --temperature 0.6 --seed $SEED > logs/infer_$(basename $T)_test.out 2>&1 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
echo "val exit $r1 test exit $r2 $(date)"
[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && echo "INFER_DONE $(date)" || echo "INFER_FAIL $(date)"
