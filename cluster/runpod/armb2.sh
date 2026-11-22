#!/bin/bash
# armb2.sh: wait for B2_EXIT in /root/pipe_a.log; on 0 run fixvt.sh on the final B2 checkpoint (seed 44), else log INFER_FAIL.
exec >> /root/infer_edge_b2.log 2>&1
until grep -q "B2_EXIT" /root/pipe_a.log; do sleep 30; done
grep -q "B2_EXIT 0" /root/pipe_a.log || { echo "INFER_FAIL upstream $(grep B2_EXIT /root/pipe_a.log) $(date)"; exit 1; }
M=weights/edge_ft_b2/$(cd /path/to/cluster/jepa-nureasoning/weights/edge_ft_b2 && ls -d update-* | sort -t- -k2 -n | tail -1)
echo "== armb2: final checkpoint $M $(date)"
exec /root/fixvt.sh $M runs/edge_b2 44
