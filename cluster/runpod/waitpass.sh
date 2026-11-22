#!/bin/bash
# waitpass.sh EXPECTED NSH PASS... : wait for the fetch shards of these passes to exit, then re-run any pass whose
# marker count is short (plan_fetch resumes, skipping done clips); up to 4 rounds. Exit 0 when every pass is complete.
D=/path/to/cluster/jepa-nureasoning/data/img_edge/train
EXP=$1; NSH=$2; shift 2
# When called for the r2 passes (pod A, GPU idle between B1 and B2), also fetch val/test (tag vt) in the same window
# and wait for it, so B2 never shares the pod's CPU quota with fetchers.
VTD=/path/to/cluster/jepa-nureasoning/data/img_edge
vtcnt() { ls $VTD/validation/*/_done_vt.json $VTD/test/*/_done_vt.json 2>/dev/null | wc -l; }
if [[ " $* " == *":r2 "* || " $* " == *":r2" ]] && [ $(vtcnt) -lt 2000 ] && ! pgrep -f "tag vt" > /dev/null; then
  cd /path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning
  for i in $(seq 0 11); do HF_HOME=$N/.cache/hf PYTHONNOUSERSITE=1 nohup nice -n 19 $N/ext/edge_env/bin/python -I -u src/edge_fetch.py --maxpix 1306624 --tag vt --splits validation,test --shard $i --nshards 12 --clips 4 --K 12 > logs/fetch_vt_$i.out 2>&1 < /dev/null & done
  echo "vt fetch launched $(date)"
fi
cnt() { c=${1%%:*}; t=${1#*:}; [ "$t" = "$1" ] && t=""; ls $D/*/_done_${c}${t:+_$t}.json 2>/dev/null | wc -l; }
for round in 1 2 3 4 5; do
  while pgrep -f "edge_fetch.py" > /dev/null && [ "$(for p in "$@"; do c=${p%%:*}; t=${p#*:}; [ "$t" = "$p" ] && t=""; pgrep -f "cams $c ${t:+--tag $t }--splits" ; done | wc -l)" -gt 0 ]; do sleep 30; done
  short=(); for p in "$@"; do n=$(cnt $p); echo "round $round pass $p markers $n / $EXP $(date)"; [ $n -lt $EXP ] && short+=($p); done
  if [ ${#short[@]} -eq 0 ]; then
    while pgrep -f "tag vt" > /dev/null; do sleep 30; done
    [[ " $* " == *":r2"* ]] && echo "vt markers $(vtcnt) / 2000"
    echo "PASSES_COMPLETE $* $(date)"; exit 0
  fi
  /root/fetch.sh $NSH "${short[@]}"; sleep 20
done
echo "PASSES_INCOMPLETE $* $(date)"; exit 1
