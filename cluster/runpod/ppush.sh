#!/bin/bash
# ppush.sh HOST PORT LIST TAG N : split LIST into N chunks and push them in parallel (each its own tar | ssh).
H=$1; PT=$2; L=$3; T=$4; NP=$5; cd ~/rp
split -n l/$NP -d -a 2 $L chunk_${T}_
for f in chunk_${T}_*; do setsid nohup ./push.sh $H $PT $f ${T}_${f##*_} > push_${T}_${f##*_}.log 2>&1 < /dev/null & done
sleep 1; echo launched $(ls chunk_${T}_* | wc -l)
