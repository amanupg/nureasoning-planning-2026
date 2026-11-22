#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_avla_setup
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/avla_setup_%j.out
# AutoVLA (UCLA, Academic Software License, accepted by Aman 2026-10-23): env .venv-avla pinned to its requirements
# (torch 2.4.0 cu121, transformers 4.49.0, qwen-vl-utils 0.0.10, lightning 2.2.1; nuplan-devkit not needed for inference),
# base Qwen2.5-VL-3B-Instruct + AutoVLA_PDMS_89.ckpt into weights/autovla (public repos, no token), then a load check.
source /etc/profile
D=/path/to/cluster/jepa-nureasoning; cd $D
export UV_CACHE_DIR=$D/.uv-cache UV_PYTHON_INSTALL_DIR=$D/.uv-python PYTHONNOUSERSITE=1 HF_HOME=$D/.hf-avla HF_HUB_DISABLE_TELEMETRY=1
UV=uv; P=$D/.venv-avla/bin/python
[ -x $P ] || $UV venv --python 3.12 $D/.venv-avla || exit 1
$UV pip install --python $P --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
    "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" "transformers==4.49.0" "tokenizers==0.21.1" "accelerate==1.5.2" peft "qwen-vl-utils==0.0.10" \
    "sentencepiece==0.2.1" "safetensors==0.5.3" "pytorch-lightning==2.2.1" "lightning-utilities==0.14.2" "numpy<2" pyyaml opencv-python-headless pillow huggingface_hub "setuptools<75" || exit 1
mkdir -p $D/weights/autovla
$D/.venv-avla/bin/huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir $D/weights/autovla/Qwen2.5-VL-3B-Instruct || exit 1
$D/.venv-avla/bin/huggingface-cli download Zewei-Zhou/AutoVLA AutoVLA_PDMS_89.ckpt --local-dir $D/weights/autovla || exit 1
du -sh $D/weights/autovla/*; rm -rf $HF_HOME $UV_CACHE_DIR
cd $D/ext/AutoVLA && $P - <<'PY'
import torch, yaml, sys
sys.path.insert(0, ".")
from models.autovla import AutoVLA
cfg = yaml.safe_load(open("config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
cfg["model"]["pretrained_model_path"] = "../../weights/autovla/Qwen2.5-VL-3B-Instruct"
cfg["inference"]["sample"] = {"max_length": 2048, "temperature": 0.01, "top_k": 0, "top_p": 1.0}
m = AutoVLA(cfg, inference=True, device="cpu")
sd = torch.load("../../weights/autovla/AutoVLA_PDMS_89.ckpt", map_location="cpu", weights_only=False)
print("ckpt keys", list(sd)[:5]); sd = sd["state_dict"]
sd = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in sd.items()}
mi, un = m.load_state_dict(sd, strict=False)
print("tensors", len(sd), "missing", len(mi), mi[:5], "unexpected", len(un), un[:5])
PY
echo AVLA_SETUP_DONE
