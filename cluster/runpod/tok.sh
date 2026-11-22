#!/bin/bash
# tok.sh HOST PORT : copy the HF token file to the pod at the same path, mode 600, never printed or in argv.
H=$1; PT=$2; T=/path/to/cluster/nureasoning/.cache/hf
cat $T/token | ssh -i $HOME/.ssh/runpod_nr -o IdentitiesOnly=yes -o BatchMode=yes -p $PT root@$H "umask 077; mkdir -p $T; cat > $T/token; chmod 600 $T/token; stat -c '%a %s' $T/token"
