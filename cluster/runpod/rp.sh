#!/bin/bash
# rp.sh HOST PORT CMD... : run a command on a RunPod pod with the dedicated key
H=$1; PT=$2; shift 2
exec ssh -i $HOME/.ssh/runpod_nr -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $PT root@$H "$@"
