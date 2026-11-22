#!/bin/bash
# mon.sh : watch both pods every 3 min; exit (to wake Claude) on any stall, error, unreachable pod, or new milestone.
# ACK env: acknowledged milestones, e.g. "a:PASSES_COMPLETE rlstart b1start".
S=$(cd "$(dirname "$0")" && pwd)   # rp.sh lives next to this script
L=/path/to/cluster/jepa-nureasoning/logs
last_a=; last_b=; same_a=0; same_b=0; down_a=0; down_b=0
while true; do
  for x in ${PODS:-"a:HOST_A:PORT_A b:HOST_B:PORT_B"}; do set -- $(echo $x | tr : " "); k=$1
    st=$($S/rp.sh $2 $3 "/root/fstat.sh; tail -n 2 /root/pipe_$k.log 2>/dev/null | tr '\n' ' '; echo; for f in $L/rp_b1.out $L/rp_b2.out $L/rp_rl.out; do [ -f \$f ] && echo \"\$(basename \$f) \$(grep -c -E 'update [0-9]+/' \$f) upd, tb \$(grep -c Traceback \$f), last: \$(grep -E 'update [0-9]+/' \$f | tail -1 | cut -c1-90)\"; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; for f in /root/infer_*.log; do [ -f \$f ] && echo \"INF \$(basename \$f) \$(tail -n 1 \$f | cut -c1-80)\"; done; for f in \$(ls -t $L/infer_*_*.out 2>/dev/null | head -2); do echo \"IP \$(basename \$f) \$(grep -o -E '[0-9]+/1000 [0-9.]+ s/clip' \$f | tail -1)\"; done" 2>/dev/null)
    if [ -z "$st" ]; then eval "down_$k=\$(( down_$k + 1 ))"; eval d=\$down_$k; echo "$(date +%H:%M) pod $k UNREACHABLE ($d)"; [ $d -ge 2 ] && { echo "ALERT pod $k unreachable twice"; exit 0; }; continue; fi
    eval "down_$k=0"; echo "$(date +%H:%M) pod $k | $(echo "$st" | tr '\n' '|')"
    echo "$st" | grep -q -E "err [1-9]|'failed': [1-9]|PASSES_INCOMPLETE|_EXIT [1-9]|tb [1-9]" && { echo "ALERT pod $k problem"; exit 0; }
    echo "$st" | grep "^INF " | grep -q "INFER_FAIL" && { echo "ALERT pod $k inference failed"; exit 0; }
    for m in $(echo "$st" | grep "^INF " | grep "INFER_DONE" | awk '{print $2}'); do [[ " $ACK " == *" $k:$m "* ]] || { echo "MILESTONE pod $k INFER_DONE $m"; exit 0; }; done
    ip=$(echo "$st" | grep "^IP " | tr '\n' ' '); eval pip=\$lastip_$k
    if [ -n "$ip" ] && [ "$ip" = "$pip" ] && ! echo "$st" | grep "^INF " | grep -q -E "INFER_(DONE|FAIL)"; then eval "sameip_$k=\$(( sameip_$k + 1 ))"; else eval "sameip_$k=0"; fi
    eval si=\$sameip_$k; [ "${si:-0}" -ge 2 ] && { echo "ALERT pod $k inference stalled 6+ min: $ip"; exit 0; }; eval "lastip_$k=\$ip"
    for m in B1_EXIT B2_EXIT RL_EXIT PASSES_COMPLETE; do echo "$st" | grep -q "$m" && ! [[ " $ACK " == *" $k:$m "* ]] && { echo "MILESTONE pod $k $m"; exit 0; }; done
    for lg in b1 b2 rl; do echo "$st" | grep -q -E "rp_$lg.out [1-9][0-9]* upd" && ! [[ " $ACK " == *" ${lg}start "* ]] && { echo "MILESTONE pod $k $lg training logging"; exit 0; }; done
    sig=$(echo "$st" | head -1); procs=$(echo "$sig" | awk '{print $2}'); eval prev=\$last_$k
    if [ "$sig" = "$prev" ] && [ "${procs:-0}" -gt 0 ]; then eval "same_$k=\$(( same_$k + 1 ))"; else eval "same_$k=0"; fi
    eval s=\$same_$k; [ $s -ge 2 ] && { echo "ALERT pod $k fetch stalled 6+ min: $sig"; exit 0; }
    eval "last_$k=\$sig"
  done
  sleep 180
done
