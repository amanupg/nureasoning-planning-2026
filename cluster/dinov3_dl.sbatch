#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_dino_dl
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/dinov3_dl_%j.out
# DINOv3 ViT-B/16 (gated on HF) into our folder. Token READ from the main track's HF token file via HF_TOKEN_PATH
# (never on a command line); download cache in our folder, removed afterwards.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export HF_TOKEN_PATH=/path/to/cluster/nureasoning/.cache/hf/token HF_HOME=$D/.hf-dino HF_HUB_DISABLE_TELEMETRY=1 PYTHONNOUSERSITE=1
$D/.venv-qd/bin/hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir $D/weights/dinov3-vitb16 || exit 1
ls -la $D/weights/dinov3-vitb16; rm -rf $HF_HOME
$D/.venv-qd/bin/python -c "
from transformers import AutoModel; m = AutoModel.from_pretrained('$D/weights/dinov3-vitb16'); c = m.config
print(type(m).__name__, 'hidden', c.hidden_size, 'registers', getattr(c, 'num_register_tokens', 0), 'patch', c.patch_size, 'params %.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6))"
echo DINO_DL_DONE
