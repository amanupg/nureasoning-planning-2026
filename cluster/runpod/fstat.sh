#!/bin/bash
# fstat.sh : fetch progress on this pod
D=/path/to/cluster/jepa-nureasoning/data/img_edge/train
echo "procs $(pgrep -f edge_fetch.py | wc -l) F $(ls $D/*/_done_F.json 2>/dev/null | wc -l) other $(ls $D/*/_done_other.json 2>/dev/null | wc -l) F_r2 $(ls $D/*/_done_F_r2.json 2>/dev/null | wc -l) other_r2 $(ls $D/*/_done_other_r2.json 2>/dev/null | wc -l) vt $(ls $D/../validation/*/_done_vt.json $D/../test/*/_done_vt.json 2>/dev/null | wc -l) GB $(du -sm $D/.. 2>/dev/null | awk '{printf "%.1f", $1/1024}') failed $(grep -h -o "'failed': [1-9][0-9]*" /path/to/cluster/jepa-nureasoning/logs/fetch_*.out 2>/dev/null | tail -1) err $(grep -l -E 'Traceback|HTTP Error 429|Too Many Requests| 429 ' /path/to/cluster/jepa-nureasoning/logs/fetch_*.out 2>/dev/null | wc -l)"
