#!/bin/bash
# fetch.sh NSH PASS... : PASS = F | other | F:r2 | other:r2. Fetch train images from HF on this pod, NSH shards per pass.
D=/path/to/cluster/jepa-nureasoning; N=/path/to/cluster/nureasoning; cd $D
export HF_HOME=$N/.cache/hf PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
NSH=$1; shift; mkdir -p logs
for p in "$@"; do
  c=${p%%:*}; t=${p#*:}; [ "$t" = "$p" ] && t=""
  mp=735000; [ "$c" = other ] && mp=98304
  if pgrep -f "cams $c ${t:+--tag $t }--splits" > /dev/null; then echo "pass $p already running, skip"; continue; fi
  for i in $(seq 0 $((NSH-1))); do
    nohup nice -n 19 $N/ext/edge_env/bin/python -I -u src/edge_fetch.py --maxpix $mp --cams $c ${t:+--tag $t} --splits train \
      --shard $i --nshards $NSH --clips 6 --K 12 > logs/fetch_${c}${t:+_$t}_$i.out 2>&1 < /dev/null &
  done
done
sleep 2; echo launched $(pgrep -f edge_fetch.py | wc -l)
