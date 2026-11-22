#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_qd_setup
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/qd_setup_%j.out
# Qwen-Drive-1.0-4B (Apache-2.0): separate env .venv-qd (pinned per ext/Qwen-Drive-1.0/requirements.txt, torch CUDA 12.8
# build) so the WA-JEPA env is untouched; weights into weights/qwen-drive (public repo, no token). flash-attn is
# skipped (sdpa is supported); causal-conv1d is best effort (transformers falls back to torch if missing).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export UV_CACHE_DIR=$D/.uv-cache UV_PYTHON_INSTALL_DIR=$D/.uv-python PYTHONNOUSERSITE=1 HF_HOME=$D/.hf-qd HF_HUB_DISABLE_TELEMETRY=1
UV=uv; P=$D/.venv-qd/bin/python
if [ ! -x $P ]; then $UV venv --python 3.12 $D/.venv-qd || exit 1; fi
$UV pip install --python $P --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" "transformers==5.14.1" "accelerate==1.12.0" \
    "safetensors==0.8.0" "numpy==2.2.6" "pillow==12.0.0" opencv-python-headless "matplotlib==3.10.7" "tqdm==4.67.1" \
    "flash-linear-attention==0.5.1" huggingface_hub ninja packaging || exit 1
$UV pip install --python $P --no-deps -e $D/ext/Qwen-Drive-1.0 || exit 1
$UV pip install --python $P --no-build-isolation "causal-conv1d==1.6.2.post1" || echo "causal-conv1d not installed (torch fallback)"
$P -c "import torch, transformers, qwen_drive; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__)" || exit 1
$D/.venv-qd/bin/hf download Qwen/Qwen-Drive-1.0-4B --local-dir $D/weights/qwen-drive || exit 1
find $D/weights/qwen-drive -maxdepth 2 | grep -v "/\.cache" | head -40; du -sh $D/weights/qwen-drive $D/.venv-qd
rm -rf $UV_CACHE_DIR $HF_HOME
echo QD_SETUP_DONE
