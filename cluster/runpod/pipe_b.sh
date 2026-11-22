#!/bin/bash
# Pod B pipeline: RL clips F+other complete -> RL training.
exec >> /root/pipe_b.log 2>&1
/root/waitpass.sh 4633 12 F other || exit 1
bash /root/runrl.sh
