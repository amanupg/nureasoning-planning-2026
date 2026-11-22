#!/bin/bash
# push.sh HOST PORT LIST TAG : stream the paths in LIST (relative to /) to a RunPod pod, mirrored at the same absolute paths.
H=$1; PT=$2; L=$3; T=$4
SSH="ssh -i $HOME/.ssh/runpod_nr -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 -p $PT root@$H"
echo "start $(date)"
nice -n 10 tar cf - -C / -T $L | $SSH "tar xf - -C / --no-same-owner" && echo "PUSH_DONE $T $(date)" || echo "PUSH_FAIL $T $(date)"
