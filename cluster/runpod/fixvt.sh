#!/bin/bash
# fixvt.sh MODEL TAG SEED: mark vt-fetched val/test dirs with _done.json, run test inference at once, fetch the 31-clip
# val dup extras (_done_dup.json), then run val (dupfix) inference. Logs /root/infer_<tag>.log, INFER_DONE/INFER_FAIL.
M=$1; T=$2; SEED=$3
D=/path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning; cd $D
exec >> /root/infer_$(basename $T).log 2>&1
n=0; for f in data/img_edge/validation/*/_done_vt.json data/img_edge/test/*/_done_vt.json; do g=$(dirname $f)/_done.json; [ -f $g ] || { cp $f $g; n=$((n+1)); }; done
echo "== fixvt: $n _done.json markers added $(date)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p runs logs; rm -f runs/$(basename $T)_test_shard0of1.* runs/$(basename $T)_dupfix_validation_shard0of1.*
HF_HUB_OFFLINE=1 $N/ext/edge_env/bin/python -I -u src/edge_infer.py --model $M --split test --out $T --shard 0/1 --K 6 --temperature 0.6 --seed $SEED > logs/infer_$(basename $T)_test.out 2>&1 & p2=$!
HF_HOME=$N/.cache/hf OMP_NUM_THREADS=2 nice -n 19 $N/ext/edge_env/bin/python -I -u src/edge_fetch.py --shard 0 --nshards 1 --clips 4 --K 12 --splits validation --members data/edge_members_valdup.json --done-name _done_dup.json > logs/rp_valdup_fetch.out 2>&1
echo "valdup fetch exit $? dup markers $(ls data/img_edge/validation/*/_done_dup.json 2>/dev/null | wc -l) $(date)"
HF_HUB_OFFLINE=1 $N/ext/edge_env/bin/python -I -u src/edge_infer.py --model $M --split validation --inputs _dupfix --out $T --shard 0/1 --K 6 --temperature 0.6 --seed $SEED > logs/infer_$(basename $T)_val.out 2>&1 & p1=$!
wait $p1; r1=$?; wait $p2; r2=$?
s1=$(grep -o "EDGE_INFER_DONE [0-9]* clips ([0-9]* skipped" logs/infer_$(basename $T)_val.out); s2=$(grep -o "EDGE_INFER_DONE [0-9]* clips ([0-9]* skipped" logs/infer_$(basename $T)_test.out)
echo "val exit $r1 [$s1] test exit $r2 [$s2] $(date)"
[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && echo "$s1" | grep -q "1000 clips (0 skipped" && echo "$s2" | grep -q "1000 clips (0 skipped" && echo "INFER_DONE $(date)" || echo "INFER_FAIL $(date)"
