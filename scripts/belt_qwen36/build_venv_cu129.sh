#!/bin/bash
# Build the generation venv for hosts whose NVIDIA driver only supports CUDA
# 12.9 (Artem-8: driver 575). Same path and package versions as the CUDA 13
# venv on Artem-3; only torch and vLLM come as their cu129 builds.
#   ./build_venv_cu129.sh <reqs_common.txt>
# reqs_common.txt: `pip freeze` of the working venv without torch*, vllm,
# flashinfer, triton, nvidia-*, cupy, flash-linear-attention.
set -e
source /mnt/data/users/vlad.smirnov/belt_qwen36/env.sh
VENV=/mnt/data/users/vlad.smirnov/venvs/belt-qwen36
CU=https://download.pytorch.org/whl/cu129
VLLM=https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
rm -rf $VENV
python3 -m venv $VENV
P=$VENV/bin/pip
$P install -q --upgrade pip
$P install -q torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url $CU
$P install -q "$VLLM" --extra-index-url $CU
$P install -q --no-deps -r "$1"
$P install -q --no-deps flash-linear-attention==0.4.1
# vLLM pulls torchcodec 0.17, which has no cu129 build and fails to load on a
# CUDA 12 driver; prompts are text only and vLLM imports fine without it.
$P uninstall -y -q torchcodec
$VENV/bin/python - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "vllm", vllm.__version__, "transformers", transformers.__version__)
x = torch.ones(2, device="cuda"); print("gpu ok", float(x.sum()))
EOF
echo VENV_DONE
