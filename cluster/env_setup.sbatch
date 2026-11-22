#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_env
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/env_setup_%j.out
# Python env for the WA-JEPA fine-tune, entirely inside our own folder (uv cache + python too).
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export UV_CACHE_DIR=$D/.uv-cache UV_PYTHON_INSTALL_DIR=$D/.uv-python PYTHONNOUSERSITE=1
UV=uv
rm -rf $D/.venv; $UV venv --python 3.12 $D/.venv || exit 1
# ONE resolve with torch pinned to the CUDA 12.8 build: a separate later install silently upgraded torch to
# the default PyPI build (CUDA 13), which this cluster's driver cannot run (job 13243218 fell back to CPU).
$UV pip install --python $D/.venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch==2.11.0+cu128" \
    "numpy<2.6" opencv-python-headless omegaconf einops "timm==0.9.16" "accelerate>=0.30" pyquaternion torchmetrics \
    safetensors pyyaml pillow scipy huggingface_hub tqdm rich || exit 1
$D/.venv/bin/python -c "import torch, timm, cv2, omegaconf; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'timm', timm.__version__)"
rm -rf $UV_CACHE_DIR; du -sh $D/.venv $D/.uv-python
echo ENV_DONE
