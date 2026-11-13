#!/usr/bin/env bash
# One-time environment for the CT -> 3D anatomy demo (Ubuntu, RTX 50xx, CUDA 13).
set -euo pipefail
cd "$(dirname "$0")"
UV=${UV:-$HOME/.local/bin/uv}
[ -x .venv/bin/python ] || $UV venv -p 3.12 .venv
PY=.venv/bin/python
$UV pip install -p $PY torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
$UV pip install -p $PY "monai==1.6.1" nibabel vtk trimesh remotezip huggingface_hub safetensors \
    einops scipy pandas tqdm pillow \
    tensorrt-cu13==11.4.0.106 "nvidia-modelopt[onnx]==0.47.0" onnx==1.21.0 polygraphy==0.53.6 onnxruntime-gpu==1.24.4
$PY -c "import torch,monai,tensorrt;print('torch',torch.__version__,torch.cuda.is_available(),'monai',monai.__version__,'trt',tensorrt.__version__)"
