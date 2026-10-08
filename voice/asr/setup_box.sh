#!/bin/bash
# Fresh molab box -> ready for voice/asr: repos, one NeMo venv, numba CUDA (RNNT loss), data libs.
#   curl -s https://raw.githubusercontent.com/IsNoobgrammer/BiBo/main/voice/asr/setup_box.sh | bash
# The HF token is placed separately (never in git): ~/.cache/huggingface/token.
set -euo pipefail
W=/home/marimo/work; V=/home/marimo/asrenv
mkdir -p $W/asr && cd $W
[ -d BiBo ] || git clone -q https://github.com/IsNoobgrammer/BiBo.git
[ -d NeMo ] || git clone -q --depth 1 https://github.com/NVIDIA/NeMo.git
# reuse the box's CUDA torch from system site-packages instead of downloading another one
[ -x $V/bin/python ] || uv venv -q --system-site-packages --python "$(command -v python3)" $V
uv pip install -q --python $V/bin/python -e "$W/NeMo[asr]" "transformers>=4.53,<5"   # else uv backtracks to 4.12 (tokenizers build fails on py3.13)
# NeMo's RNNT loss JIT-compiles numba CUDA kernels: needs libnvvm (numba-cuda cu13 wheels); numba-cuda 0.30 breaks on numpy>=2.4
uv pip install -q --python $V/bin/python "numba-cuda[cu13]" "numpy<2.4" pyarrow soundfile huggingface_hub wandb torchvision torchaudio  # tv/ta must match the venv torch (system ones are for 2.11)
$V/bin/python -c "import torch, nemo.collections.asr, numba.cuda; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'nemo ok')"
echo SETUP_DONE
