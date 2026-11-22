#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J jn_edge_ft
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:R6000:1
#SBATCH --exclude=ins058,ins082,ins084
#SBATCH -t 1:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/edge_train_%j.out
# Alpamayo 2 Edge fine-tune (src/edge_train.py) on one 96 GB RTX PRO 6000: full fine-tune of 2.65B with fp32 master
# weights + AdamW states is ~42 GB before activations (48 GB cards do not fit).
# MODE=smoke (default): label check + SMOKE steps on one repeated validation row (correctness only).
# MODE=smoketrain: the same on train rows (ARGS passes --cot ...).
# MODE=resumetest: 4-update mini run on validation rows into a scratch dir, stopped at update 2 and resumed (tests
#   state.pt); the scratch dir is removed afterwards. Validation rows are never used for real training.
# MODE=speedtest: UPD (40) updates of 4 validation rows, logging encode vs forward/backward time; nothing kept.
# MODE=train: the real run on data/edge_rows_train.json. Submit with -t 12:00:00. It saves state every hour and
#   stops itself before the time limit (exit 3); this script then requeues the same job id, which resumes.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
N=/path/to/cluster/nureasoning
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== attempt $(date) host $(hostname) mode ${MODE:-smoke} restarts ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
# NGPU>1: data parallel + sharded AdamW on NGPU cards of one node (submit with matching --gres and more -c/--mem).
# Host RAM: NGPU=8 needs >80G (8 ranks load fp32 weights at once; 80G OOM-killed C-2ep and C at load, 11-16); use
# -c 32 --mem=200G (MaxMemPerCPU 6400 MB ties memory to CPUs). sacct MaxRSS (~51 GB for D) misses the load peak.
if [ ${NGPU:-1} -gt 1 ]; then
  PY="$N/ext/edge_env/bin/python -I -u -m torch.distributed.run --standalone --nproc_per_node=$NGPU src/edge_train.py --model ${MODEL:-$N/models/alpamayo2_edge}"
else
  PY="$N/ext/edge_env/bin/python -I -u src/edge_train.py --model ${MODEL:-$N/models/alpamayo2_edge}"
fi
stopped() { [ -f "$1/STOPPED_FOR_REQUEUE" ] && rm -f "$1/STOPPED_FOR_REQUEUE"; }   # multi-GPU planned stop
case ${MODE:-smoke} in
  smoke)
    $PY --rows data/edge_inputs_validation.json --split validation --smoke ${SMOKE:-8} ${ARGS:-}; rc=$? ;;
  smoketrain)   # smoke on train rows (e.g. ARGS="--cot data/edge_cot_train.json --cot-only")
    $PY --rows data/edge_rows_train.json --split train --smoke ${SMOKE:-8} ${ARGS:-}; rc=$? ;;
  resumetest)
    T=$D/weights/edge_resumetest_$SLURM_JOB_ID; rm -rf "$T"
    $PY --rows data/edge_inputs_validation.json --split validation --out "$T" --accum ${ACC:-2} --max-updates 4 \
      --stop-after-updates 2 --save-every 1000 ${ARGS:-}; r1=$?; stopped "$T" && r1=3
    $PY --rows data/edge_inputs_validation.json --split validation --out "$T" --accum ${ACC:-2} --max-updates 4 \
      --save-every 1000 ${ARGS:-}; r2=$?
    ls -la "$T"; rm -rf "$T"
    echo "first run exit $r1 (want 3), resumed run exit $r2 (want 0)"
    [ $r1 -eq 3 ] && [ $r2 -eq 0 ] && echo RESUMETEST_PASS || echo RESUMETEST_FAIL; rc=0 ;;
  speedtest)
    T=$D/weights/edge_speedtest_$SLURM_JOB_ID; rm -rf "$T"
    $PY --rows ${ROWS:-data/edge_inputs_validation.json} --split ${SPLIT:-validation} --out "$T" --accum 4 --max-updates ${UPD:-40} \
      --log-every 5 --save-every 100000 --state-every-min 100000 ${ARGS:-}; rc=$?; rm -rf "$T" ;;
  train)
    # Stop ~40 min before the limit so the state save (~32 GB) finishes in time.
    LIM=$(squeue -h -j $SLURM_JOB_ID -o %l); MIN=$(echo "$LIM" | awk -F'[-:]' '{if(NF==4)print $1*1440+$2*60+$3; else if(NF==3)print $1*60+$2; else print $1}')
    $PY --rows ${ROWS:-data/edge_rows_train.json} --split train --out ${OUT:-weights/edge_ft} \
      --stop-after-min $((MIN - 40)) ${ARGS:-}; rc=$?; stopped ${OUT:-weights/edge_ft} && rc=3
    if [ $rc -eq 3 ]; then echo "requeue for resume"; scontrol requeue $SLURM_JOB_ID; fi ;;
esac
echo "EDGE_TRAIN_EXIT $rc"
